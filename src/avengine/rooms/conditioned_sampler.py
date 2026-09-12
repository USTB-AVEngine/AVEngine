"""Fixed-profile QA sampling over existing navigation and prepared sounds.

This module writes neutral plans only. Native execution and answerability are
separate: planned geometry never becomes an achieved-condition assertion.
"""
from __future__ import annotations

from avengine.qa.unified_catalog import QA_IDS

from collections import Counter
from copy import deepcopy
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.capture.neutral_readback import COORDINATE_FRAME
from avengine.qa.answerability import line_of_sight
from avengine.rooms.walkable_space import camera_grid
from avengine.routes.trajectory import resample_polyline_by_arc_length

POLICY = 'conditioned_static_v2'
CAPABILITY_REVISION = 'p05_conditions_20260910'
RIGID = {'rigid_object', 'rigid_static_object'}
SAME_FLOOR_Y_TOLERANCE_M = 0.3
DEFAULT_DISTANCE_RANGE_M = (1.5, 4.5)
CLIP_SPAN_FIT_POLICY = 'filter_to_remaining_budget_then_uniform'
SEPARATION_TARGET_ANY_LEGAL = 'any_legal_in_bin'
SEPARATION_TARGET_UNIFORM_IN_BIN = 'uniform_in_bin'
SEPARATION_TARGET_POLICIES = {SEPARATION_TARGET_ANY_LEGAL, SEPARATION_TARGET_UNIFORM_IN_BIN}

# Every enumerated planning knob this sampler reads, with the values it accepts.
# ``resolve_condition_profile`` validates against this table and
# ``describe_generator_capabilities`` publishes it, so a knob cannot be
# advertised without being read, or read without being advertised.
ENUM_KNOB_VALUES = {
    # in_fov and off_screen state one visibility for the whole audible window.
    # visible_then_hidden is the QA-25 AV shape: the same audible event has to
    # carry a visible frame that identifies the emitter and, later in that same
    # event, a hidden frame the question is asked at. A uniform mask cannot say
    # that, which is why the AV subset had no way to be planned.
    'anchor_visibility': ('in_fov', 'off_screen', 'visible_then_hidden'),
    'competitor_visibility': ('in_fov', 'off_screen'),
    'anchor_line_of_sight': ('clear', 'occluded'),
    'speech_motion': ('speaker_moving', 'competitor_moving', 'all_still'),
    # Stated, never drawn: under speaker_moving the competitors used to receive a
    # coin-flip moving flag, which let a competitor share the target's answer.
    'competitor_motion': ('still', 'moving', 'any'),
    'event_relation': ('sequential', 'overlap', 'repeat'),
    # A field-of-view crossing of the body proxy. A pixel occlusion state is a
    # different measurement and is not claimed here.
    'visibility_transition': ('none', 'out_of_view_to_visible', 'visible_then_hidden'),
}
# These values are supplied by the P03 motion solver and consumed below when
# it turns each compiled requirement into one timed route. None means the
# current request had no such motion condition. motion_window_placement is
# intentionally absent: generation_conditions.PLANNING_KEY_LAYER marks it as
# derived for recipe-conflict detection, and the solver consumes that compiled
# value directly rather than treating it as a caller-adjustable knob.
def _pixel_occlusion_values():
    """The occlusion values this build really supports, read from the solver.

    Two lists drifted apart here: ``conditioned_visibility`` implements five
    states and this table named two, so QA-08 and QA-24 emitted
    ``fully_occluded`` and QA-10 emitted ``registered_occluder_visible`` and
    both were refused at profile resolution while the solver that would have
    served them sat one call away. Reading the solver's own map means a value
    cannot be advertised here without a requirement behind it, or implemented
    there without being reachable from a request.
    """
    from avengine.rooms.conditioned_visibility import _TRANSITION_TO_REQUIREMENT

    return tuple(sorted(_TRANSITION_TO_REQUIREMENT))


SOLVER_KNOB_VALUES = {
    'target_moved_after_sound': (True, False),
    'distance_trend_during_event': ('nearer', 'farther'),
    # Filled from conditioned_visibility below, once that module is importable.
    'pixel_occlusion_transition': (),
}
SOLVER_KNOBS = tuple(SOLVER_KNOB_VALUES)
# The remaining knobs this sampler reads: counts, budgets, ranges and policies.
SCALAR_KNOBS = ('anchor_count', 'distance_range_m', 'min_gap_between_audible_windows_s',
                'minimum_overlap_s', 'reserve_tail_s', 'retry_budget_within_profile',
                'separation_bin_deg', 'separation_target_policy',
                # Which declared instance must own the earliest audible window. A
                # question whose gold answer is "who spoke first" cannot be served
                # by a uniform draw over feasible actor orders.
                'first_speaker_instance_id')
DECLARED_SAMPLER_KNOBS = tuple(sorted(
    set(ENUM_KNOB_VALUES) | set(SCALAR_KNOBS) | set(SOLVER_KNOBS)))


def _fill_pixel_occlusion_values():
    if not SOLVER_KNOB_VALUES['pixel_occlusion_transition']:
        SOLVER_KNOB_VALUES['pixel_occlusion_transition'] = _pixel_occlusion_values()
    return SOLVER_KNOB_VALUES['pixel_occlusion_transition']
# One solved neutral plan is handed to whichever room-family executor owns the
# room, so a knob solved here is solved for every route.
SUPPORTED_BACKENDS = ('spear_unreal', 'spear_usd', 'habitat')
COMPETITOR_MOTION_DEFAULT = 'any'
VISIBILITY_TRANSITION_DEFAULT = 'none'
ASSET_REPEAT_POLICIES = ('distinct_assets', 'allow_repeats')


def describe_generator_capabilities():
    """Declare which planning knobs this build reads, for the condition compiler.

    ``avengine.qa.generation_conditions.resolve_generator_capabilities`` accepts
    this module itself, so the compiler asks the planner instead of carrying a
    hand-maintained list that goes stale the moment a knob lands or leaves. The
    names come from the same tables ``resolve_condition_profile`` validates
    against. This describes an interface and nothing else: it never says a
    condition was planned, rendered or met.
    """
    _fill_pixel_occlusion_values()
    return {
        'source': __name__,
        'version': POLICY + '+' + CAPABILITY_REVISION,
        'knobs': list(DECLARED_SAMPLER_KNOBS),
        'backends': list(SUPPORTED_BACKENDS),
        'declared_at': ('src/avengine/rooms/conditioned_sampler.py ENUM_KNOB_VALUES and '
                        'SCALAR_KNOBS, read by resolve_condition_profile, select_entities, '
                        '_moving_flags, select_sounds, _solve_motion_conditions and '
                        'select_camera_and_schedule'),
    }


class ConditionedRequestConflict(ValueError):
    """A stated profile knob and a compiled question knob cannot both hold.

    ``resolve_condition_profile`` lets an explicit ``request['profile']`` win
    over a compiled condition, which is right when the two agree and silent
    when they do not: the Episode is then planned for a recipe that cannot
    answer the question that was asked, and the cost is only discovered after
    the render.  This refuses that pair while it is still a dictionary.
    """

    def __init__(self, conflicts):
        self.conflicts = list(conflicts)
        detail = '; '.join(self._describe(row) for row in self.conflicts)
        super().__init__('this request cannot be planned as one Episode: ' + detail)

    @staticmethod
    def _describe(row):
        knob = row.get('knob')
        if row.get('kind') == 'two_questions':
            return (f"{knob}: {row.get('wanted_by')} ask for different values "
                    f"({row.get('requested')!r}), and one Episode has one {knob}")
        if row.get('kind') == 'entity_cannot_satisfy':
            return (f"{knob}={row.get('compiled')!r} cannot be met: {row.get('requested')}. "
                    f"Asked for by {row.get('wanted_by')}")
        return (f"{knob}: the request profile states {row.get('requested')!r} but "
                f"{row.get('wanted_by')} needs {row.get('compiled')!r}")


class ConditionedPlanningFailure(ValueError):
    def __init__(self, result):
        self.result = result
        super().__init__('fixed condition profile exhausted: ' + str(result['failure_histogram']))


class CandidateFailure(ValueError):
    def __init__(self, stage, reason):
        self.stage, self.reason = stage, reason
        super().__init__(reason)


def _profile_admits(requested, compiled):
    """Whether a stated knob value still leaves the compiled value reachable.

    A scalar has to equal it. A distribution admits it when it is one of the
    published choices, because the sampler may still draw it.
    """
    if isinstance(requested, Mapping):
        choices = requested.get('choices')
        if choices is None:
            return True
        return any(choice == compiled for choice in choices)
    if isinstance(requested, (list, tuple)) and isinstance(compiled, (list, tuple)):
        return list(requested) == list(compiled)
    return requested == compiled


def _draw(value, rng):
    if not isinstance(value, Mapping):
        return deepcopy(value)
    choices = value.get('choices', value.get('bins'))
    if not isinstance(choices, list) or not choices:
        raise ValueError('condition distribution needs nonempty choices/bins')
    weights = value.get('weights')
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(choices) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError('invalid explicit condition weights')
        weights = weights / weights.sum()
    return deepcopy(choices[int(rng.choice(len(choices), p=weights))])


def histogram_separation_5deg(angles):
    """Histogram achieved (not requested) separation in closed-open 5 degree bins."""
    counts = Counter()
    values = []
    for raw in angles:
        if raw is None:
            continue
        angle = float(raw)
        if not math.isfinite(angle):
            continue
        values.append(angle)
        start = 175 if angle >= 180 else int(min(max(angle, 0.0), 179.999999) // 5) * 5
        counts[start] += 1
    bins = [{'lo_deg': lo, 'hi_deg': lo + 5, 'count': int(counts.get(lo, 0))} for lo in range(0, 180, 5)]
    return {'bin_width_deg': 5, 'unit': 'achieved_separation_deg',
            'requested_bin_is_not_coverage': True, 'count': len(values),
            'occupied_bins': [row for row in bins if row['count']], 'bins': bins}


def _parse_floor_height_token(token):
    if isinstance(token, Mapping):
        for key in ('floor_height_m', 'floor_y_m', 'height_m'):
            if token.get(key) is not None:
                return float(token[key])
        token = token.get('subroom_id') or token.get('floor_id') or token.get('id')
    if token is None:
        return None
    if isinstance(token, (int, float)) and not isinstance(token, bool):
        return float(token)
    text = str(token)
    if '_floor_' in text:
        tail = text.rsplit('_floor_', 1)[1]
        try:
            return float(tail)
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _floor_level_clusters(levels):
    clusters = []
    for value, weight in sorted(levels, key=lambda pair: pair[0]):
        if not clusters or value - clusters[-1][-1][0] > SAME_FLOOR_Y_TOLERANCE_M:
            clusters.append([])
        clusters[-1].append((float(value), int(weight)))
    return clusters


def declared_floor_heights_m(room, space=None):
    """Unique measured/planned navigable floors for a room and its space."""
    floors = []
    measured_levels = []
    package = {}
    if isinstance(room, Mapping):
        raw_package = room.get('room_package')
        package = raw_package if isinstance(raw_package, Mapping) else (
            room if isinstance(room.get('floor_reference'), Mapping) else {}
        )
        for collection in (package.get('subrooms'), room.get('subrooms'),
                           package.get('floor_heights_m'), room.get('floor_heights_m'),
                           package.get('planning_floors_m'), room.get('planning_floors_m')):
            if isinstance(collection, (list, tuple)):
                for item in collection:
                    height = _parse_floor_height_token(item)
                    if height is not None and math.isfinite(height):
                        floors.append(float(height))
        floor_reference = package.get('floor_reference')
        reference_path = (
            floor_reference.get('path')
            if isinstance(floor_reference, Mapping)
            else None
        )
        if isinstance(reference_path, str) and reference_path:
            try:
                payload = json.loads(Path(reference_path).read_text())
                levels = (payload.get('summary') or {}).get('levels', [])
                for item in levels:
                    height = _parse_floor_height_token(item)
                    if height is not None and math.isfinite(height):
                        measured_levels.append((float(height), int(
                            item.get('sample_count', 1)
                        ) if isinstance(item, Mapping) else 1))
                # Collapse noisy snap levels by tolerance, retaining the
                # measured modal level instead of the first tiny outlier.
                for cluster in _floor_level_clusters(measured_levels):
                    floors.append(max(cluster, key=lambda pair: pair[1])[0])
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
    meta = getattr(space, 'metadata', None) or {}
    if isinstance(meta, Mapping):
        extra = meta.get('floor_heights_m') or meta.get('planning_floors_m')
        if isinstance(extra, (list, tuple)):
            floors.extend(float(v) for v in extra if v is not None and math.isfinite(float(v)))
    unique = []
    for height in sorted(floors):
        if not unique or abs(height - unique[-1]) > SAME_FLOOR_Y_TOLERANCE_M:
            unique.append(height)
    return unique


def static_support_floor_reference(room, space, placement_rows):
    """Select a measured room floor for support-only static planning.

    Support Y is used only to choose among measured room levels; it is never
    published or treated as the floor height. This keeps navigation/floor
    semantics separate from tabletop and wall root transforms.
    """
    floors = declared_floor_heights_m(room, space)
    raw_package = room.get('room_package') if isinstance(room, Mapping) else None
    package = raw_package if isinstance(raw_package, Mapping) else (
        room if isinstance(room, Mapping) and isinstance(
            room.get('floor_reference'), Mapping
        ) else {}
    )
    if not floors:
        return float((getattr(space, 'metadata', {}) or {}).get(
            'floor_height_m', 0.0)), 'space_metadata.floor_height_m'
    support_ys = []
    for placement in (placement_rows or {}).values():
        try:
            support_ys.append(float(
                placement['root_transform']['translation_m'][1]
            ))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    if support_ys:
        target = float(np.median(support_ys))
        # Floor-reference snap levels with one sample are local navmesh noise;
        # retain measured modal levels when choosing the ground below a support.
        modal = []
        floor_reference = package.get('floor_reference') if isinstance(
            package, Mapping
        ) else None
        reference_path = (
            floor_reference.get('path')
            if isinstance(floor_reference, Mapping)
            else None
        )
        if isinstance(reference_path, str) and reference_path:
            try:
                payload = json.loads(Path(reference_path).read_text())
                for item in (payload.get('summary') or {}).get('levels', []):
                    if isinstance(item, Mapping) and int(item.get('sample_count', 1)) >= 2:
                        height = _parse_floor_height_token(item)
                        if height is not None and math.isfinite(height):
                            modal.append(float(height))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        floor_levels = sorted(set(modal)) or floors
        below_support = [value for value in floor_levels if value <= target - 0.2]
        if below_support:
            return max(below_support), (
                'measured_room_floor_below_static_support'
            )
        return min(floor_levels, key=lambda value: abs(float(value) - target)), (
            'measured_room_floor_nearest_static_support'
        )
    return float(floors[0]), 'measured_room_floor_default'


def lock_same_floor_region(space, rng, region=None, room=None):
    """Sample a floor first in multi-floor scenes, then lock |Delta y| to 0.3 m."""
    bounds = space.bounds().copy() if region is None else np.asarray(region, dtype=float).copy()
    floors = declared_floor_heights_m(room, space)
    if len(floors) > 1:
        floor_y = float(floors[int(rng.integers(len(floors)))])
    elif floors:
        floor_y = float(floors[0])
    else:
        hub = space.sample_navigable(rng, region)
        floor_y = float(hub[1])
    bounds[0, 1] = floor_y - SAME_FLOOR_Y_TOLERANCE_M
    bounds[1, 1] = floor_y + SAME_FLOOR_Y_TOLERANCE_M
    return bounds, floor_y


def _points_same_floor(points, floor_y=None, tolerance=SAME_FLOOR_Y_TOLERANCE_M):
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(pts):
        return True, 0.0 if floor_y is None else float(floor_y)
    ys = pts[:, 1]
    if floor_y is None:
        floor_y = float(np.median(ys))
    return bool(np.all(np.abs(ys - floor_y) <= tolerance)), float(floor_y)


def _camera_grid_on_floor(space, *, height, region, floor_y):
    saved = None
    had = isinstance(getattr(space, 'metadata', None), dict) and 'floor_height_m' in space.metadata
    if isinstance(getattr(space, 'metadata', None), dict):
        saved = space.metadata.get('floor_height_m')
        space.metadata['floor_height_m'] = float(floor_y)
    try:
        positions = camera_grid(space, step_m=.55, height_above_floor_m=height, region=region)
    finally:
        if isinstance(getattr(space, 'metadata', None), dict):
            if had:
                space.metadata['floor_height_m'] = saved
            else:
                space.metadata.pop('floor_height_m', None)
    kept = []
    for pos in positions:
        if abs(float(pos[1]) - float(height) - float(floor_y)) <= SAME_FLOOR_Y_TOLERANCE_M:
            kept.append(pos)
    return kept


def instance_requests(request):
    """Read the physical instances a request declares, without inventing any.

    A row may name ``entity_instance_id``/``instance_id``, ``asset_id``,
    ``source_class``, ``role`` and ``speaking``. Two rows may name the same
    asset: that is two physical instances of one registered asset, and their
    instance IDs stay distinct.
    """
    entities = request.get('entities') if isinstance(request, Mapping) else None
    rows = None
    for candidate in (
        (entities or {}).get('instances'),
        request.get('instances') if isinstance(request, Mapping) else None,
        request.get('entity_instances') if isinstance(request, Mapping) else None,
    ):
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            rows = candidate
            break
    if rows is None:
        return []
    result = []
    for index, row in enumerate(rows):
        if hasattr(row, 'to_dict'):
            row = row.to_dict()
        if not isinstance(row, Mapping):
            raise ValueError('each declared instance must be an object')
        instance_id = row.get('entity_instance_id') or row.get('instance_id')
        speaking = row.get('speaking')
        if speaking is not None and not isinstance(speaking, bool):
            raise ValueError('declared instance speaking must be true or false')
        # The instance may be named anything, but the slot is the name the UE and
        # RLR endpoint conventions are built on, so it stays source1..sourceN.
        slot = str(row.get('source_slot_id') or f'source{index + 1}')
        if slot != f'source{index + 1}':
            raise ValueError(
                f'instance {index + 1} asks for backend slot {slot!r}; the source endpoint '
                f'convention needs source{index + 1}. Name the instance itself instead.')
        result.append({
            'entity_instance_id': None if instance_id is None else str(instance_id),
            'asset_id': None if row.get('asset_id') is None else str(row['asset_id']),
            'source_class': None if row.get('source_class') is None else str(row['source_class']),
            'role': None if row.get('role') is None else str(row['role']),
            'speaking': speaking,
            'source_slot_id': slot,
        })
    declared_ids = [row['entity_instance_id'] for row in result if row['entity_instance_id']]
    if len(set(declared_ids)) != len(declared_ids):
        raise ValueError('declared entity_instance_id values must be distinct')
    return result


def resolve_condition_profile(request, registry, *, question_knobs=None):
    """Resolve N/S/classes and all quota choices once, before room/route retries.

    ``question_knobs`` carries the planning knobs a compiled QA condition asks
    for. An explicit ``request['profile']`` value always wins, so a question can
    drive the sampler without silently overriding what the caller stated.
    """
    rng = np.random.default_rng(int(request.get('seed', 0)))
    camera = request.get('camera', {})
    if camera.get('motion', request.get('camera_motion', 'static')) != 'static':
        raise ValueError('conditioned_static_v2 requires static camera motion')
    entities, requested_profile = request.get('entities', {}), request.get('profile', {})
    profile = dict(requested_profile)
    knob_sources = {key: 'request_profile' for key in profile}
    knob_owner = dict((question_knobs or {}).get('__wanted_by__') or {})
    conflicts = []
    for key, value in dict(question_knobs or {}).items():
        if key == '__wanted_by__':
            continue
        if (key not in ENUM_KNOB_VALUES and key not in SCALAR_KNOBS
                and key not in SOLVER_KNOB_VALUES):
            continue
        if key in requested_profile:
            # An explicit statement still wins, but only when it can actually
            # hold together with the question. A distribution is honoured when
            # the compiled value is one of its choices.
            if not _profile_admits(requested_profile[key], value):
                conflicts.append({'knob': key, 'requested': requested_profile[key],
                                  'compiled': value,
                                  'wanted_by': knob_owner.get(key, 'the compiled question')})
            continue
        profile[key] = value
        knob_sources[key] = 'compiled_question_condition'
    if conflicts:
        raise ConditionedRequestConflict(conflicts)
    explicit = request.get('source_asset_ids')
    declared = instance_requests(request)
    records = {r['asset_id']: r for r in registry['assets']}
    count_spec = entities.get('total_count', len(explicit) if explicit is not None else
                              (len(declared) if declared else 2))
    if declared and explicit is not None and len(declared) != len(explicit):
        raise ValueError('declared instances and explicit source IDs must have the same count')
    if explicit is not None:
        n = len(explicit)
        allowed = count_spec.get('choices') if isinstance(count_spec, Mapping) else [count_spec]
        if n not in allowed:
            raise ValueError('explicit source IDs must be distinct and agree with total_count')
        if any(a not in records for a in explicit):
            raise ValueError('explicit source asset is absent from the runtime registry')
        # A repeated asset ID is two instances of one asset, not a duplicate.
        classes = ['rigid_static_object' if records[a]['entity_class'] in RIGID else records[a]['entity_class'] for a in explicit]
    elif declared:
        n = len(declared)
        allowed = count_spec.get('choices') if isinstance(count_spec, Mapping) else [count_spec]
        if n not in allowed:
            raise ValueError('declared instances must agree with total_count')
        classes = []
        for row in declared:
            if row['asset_id'] is not None:
                if row['asset_id'] not in records:
                    raise ValueError('declared instance asset is absent from the runtime registry')
                record = records[row['asset_id']]
                classes.append('rigid_static_object' if record['entity_class'] in RIGID else record['entity_class'])
            elif row['source_class'] is not None:
                classes.append(row['source_class'])
            else:
                raise ValueError('a declared instance needs an asset_id or a source_class')
    else:
        n = int(_draw(count_spec, rng)); classes = None
    if not 2 <= n <= 4:
        raise ValueError('conditioned sampler requires 2 to 4 entities')
    silent_spec = entities.get('silent_count', request.get('silent_actor_count', 0))
    declared_speaking = [i for i, row in enumerate(declared) if row['speaking'] is not False]
    if declared and any(row['speaking'] is not None for row in declared):
        silent = n - len(declared_speaking)
        if 'silent_count' in entities and int(_draw(silent_spec, np.random.default_rng(0))) != silent:
            raise ValueError('declared instance speaking flags contradict silent_count')
    else:
        silent = int(_draw(silent_spec, rng))
    if not 0 <= silent < n:
        raise ValueError('silent_count must retain a speaking entity')
    minimum = int(entities.get('min_articulated_count', 0))
    if not 0 <= minimum <= n:
        raise ValueError('invalid min_articulated_count')
    if classes is None:
        class_spec = entities.get('source_classes', 'articulated_human')
        for _ in range(10000):
            classes = [_draw(class_spec, rng) for _ in range(n)]
            if sum(c != 'rigid_static_object' for c in classes) >= minimum:
                break
        else:
            raise ValueError('source class distribution cannot meet min_articulated_count')
    if any(c not in {'articulated_human', 'articulated_animal', 'rigid_static_object'} for c in classes):
        raise ValueError('unknown source class in condition profile')
    if sum(c != 'rigid_static_object' for c in classes) < minimum:
        raise ValueError('explicit sources violate min_articulated_count')
    if declared and any(row['speaking'] is not None for row in declared):
        speakers = sorted(declared_speaking)
    else:
        speakers = sorted(int(i) for i in rng.choice(n, n-silent, replace=False))
    declared_anchors = sorted(i for i, row in enumerate(declared)
                              if row['role'] in {'anchor', 'target'})
    anchor_count = int(profile.get('anchor_count', len(declared_anchors) or 1))
    if not 1 <= anchor_count <= len(speakers):
        raise ValueError('anchor_count must be within the speaking count')
    if declared_anchors:
        if any(i not in speakers for i in declared_anchors):
            raise ValueError('a declared target instance must be a speaking instance')
        if len(declared_anchors) != anchor_count:
            raise ValueError('declared target roles contradict anchor_count')
        anchors = declared_anchors
    else:
        anchors = sorted(int(i) for i in rng.choice(speakers, anchor_count, replace=False))
    sep_spec = profile.get('separation_bin_deg', {'bins': [[15.,30.],[30.,60.],[60.,90.],[90.,180.]]})
    sep = _draw(sep_spec, rng)
    floor = float(sep_spec.get('floor_deg', 15.)) if isinstance(sep_spec, Mapping) else float(profile.get('separation_floor_deg', 15.))
    if len(sep) != 2 or not 0 <= floor <= float(sep[0]) < float(sep[1]) <= 180.:
        raise ValueError('invalid separation bin or floor; lower bounds are never relaxed')
    if profile.get('competitor_set','all_other_entities_including_offscreen')!='all_other_entities_including_offscreen':
        raise ValueError('competitor_set must include every other entity')
    if profile.get('separation_window','whole_audible_window_of_anchor')!='whole_audible_window_of_anchor':
        raise ValueError('separation must hold over the whole anchor audible window')
    distance = profile.get('distance_range_m', list(DEFAULT_DISTANCE_RANGE_M))
    if not isinstance(distance, (list, tuple)) or len(distance) != 2:
        raise ValueError('distance_range_m must be [min, max] meters')
    distance = [float(distance[0]), float(distance[1])]
    if not 0 <= distance[0] < distance[1] or not all(math.isfinite(v) for v in distance):
        raise ValueError('invalid distance_range_m')
    sep_policy = profile.get('separation_target_policy', SEPARATION_TARGET_ANY_LEGAL)
    if sep_policy not in SEPARATION_TARGET_POLICIES:
        raise ValueError('unsupported separation_target_policy')
    repeat_policy = str(entities.get('asset_repeat_policy', 'distinct_assets'))
    if repeat_policy not in ASSET_REPEAT_POLICIES:
        raise ValueError('unsupported asset_repeat_policy')
    motion_request = request.get('motion') if isinstance(request.get('motion'), Mapping) else {}
    speed_range = motion_request.get('speed_range_mps') or motion_request.get('walk_speed_range_mps')
    if speed_range is None:
        speed_range = (.5, .8)
    else:
        # An explicitly requested walking speed is not replaced by a route default.
        speed_range = [float(value) for value in speed_range]
        if (len(speed_range) != 2 or not all(math.isfinite(v) for v in speed_range)
                or not 0 < speed_range[0] <= speed_range[1]):
            raise ValueError('walk speed range must be two finite positive ordered speeds')
    constructive_motion = profile.get(
        'constructive_motion', motion_request.get(
            'constructive_motion', request.get('constructive_motion', False)))
    if not isinstance(constructive_motion, bool):
        raise ValueError('constructive_motion must be a boolean')
    result = {'total_count': n, 'speaking_count': n-silent, 'silent_count': silent,
              'source_classes': classes, 'anchor_count': anchor_count, 'separation_bin_deg': list(map(float, sep)),
              'separation_floor_deg': floor, 'speaking_indices': speakers, 'anchor_indices': anchors,
              'competitor_set': 'all_other_entities_including_offscreen',
              'native_start_hold_frames': request.get('start_hold_frames',0),
              'anchor_visibility': _draw(profile.get('anchor_visibility', 'in_fov'), rng),
              'competitor_visibility': _draw(profile.get('competitor_visibility', 'in_fov'), rng),
              'anchor_line_of_sight': _draw(profile.get('anchor_line_of_sight', 'clear'), rng),
              'speech_motion': _draw(profile.get('speech_motion', 'all_still'), rng),
              'competitor_motion': _draw(profile.get('competitor_motion', COMPETITOR_MOTION_DEFAULT), rng),
              'visibility_transition': _draw(profile.get('visibility_transition', VISIBILITY_TRANSITION_DEFAULT), rng),
              'target_moved_after_sound': profile.get('target_moved_after_sound'),
              'distance_trend_during_event': profile.get('distance_trend_during_event'),
              'pixel_occlusion_transition': profile.get('pixel_occlusion_transition'),
              'event_relation': _draw(profile.get('event_relation', request.get('audio_mode', 'sequential')), rng),
              'min_gap_between_audible_windows_s': float(profile.get('min_gap_between_audible_windows_s', .5)),
              'reserve_tail_s': float(profile.get('reserve_tail_s', 3.)),
              'minimum_overlap_s': float(profile.get('minimum_overlap_s', .3)),
              'retry_budget_within_profile': int(profile.get('retry_budget_within_profile', 200)),
              'distance_range_m': distance,
              'separation_target_policy': sep_policy,
              'instances': deepcopy(declared),
              'first_speaker_instance_id': profile.get('first_speaker_instance_id'),
              # C05's measured support levels. This is a resource the caller
              # supplies, not a knob any question compiles, so it is carried
              # explicitly and stays out of the capability declaration.
              'contact_correction': deepcopy(profile.get('contact_correction')),
              # A start pin is a statement about where a declared body stands, so
              # it travels with the resolved profile. resolve_condition_profile
              # builds a fixed key set, so a knob absent from it is dropped in
              # silence, which is how this one first went missing.
              'pinned_static_positions_m': deepcopy(
                  profile.get('pinned_static_positions_m')),
              'asset_repeat_policy': repeat_policy,
              'walk_speed_range_mps': list(speed_range),
              # This is an implementation mode, not a QA condition knob. It
              # defaults to the pre-existing sampler and is recorded so a
              # plan's route construction can be read back unambiguously.
              'constructive_motion': constructive_motion,
              'knob_sources': knob_sources}
    for key, valid in ENUM_KNOB_VALUES.items():
        if result[key] not in valid:
            raise ValueError('unsupported '+key)
    _fill_pixel_occlusion_values()
    for key, valid in SOLVER_KNOB_VALUES.items():
        if result[key] is not None and result[key] not in valid:
            raise ValueError(
                'unsupported ' + key + ': ' + repr(result[key])
                + '; this build supports ' + repr(list(valid)))
    if result['speech_motion'] == 'competitor_moving' and result['competitor_motion'] == 'still':
        raise ValueError('speech_motion=competitor_moving contradicts competitor_motion=still')
    if result['competitor_motion'] == 'moving':
        competitor_classes = [classes[i] for i in range(n) if i not in anchors]
        immobile = [c for c in competitor_classes if c == 'rigid_static_object']
        mobile = [c for c in competitor_classes if c != 'rigid_static_object']
        stated_by_caller = knob_sources.get('competitor_motion') == 'request_profile'
        if stated_by_caller and immobile:
            # The caller asked for every competitor to move and one of them is a
            # registered static object. That is a real contradiction, so it is
            # refused here rather than after routes are drawn.
            raise ConditionedRequestConflict([{
                'kind': 'entity_cannot_satisfy', 'knob': 'competitor_motion',
                'requested': f'{len(immobile)} of {len(competitor_classes)} competitors are '
                             'registered static objects',
                'compiled': 'moving',
                'wanted_by': 'the request profile',
            }])
        if not stated_by_caller and not mobile:
            # A compiled question only needs one competitor whose answer differs,
            # which is what unified_catalog._p8_apply_distractor_gate reads. It is
            # unsatisfiable only when nothing in the room can move at all.
            raise ConditionedRequestConflict([{
                'kind': 'entity_cannot_satisfy', 'knob': 'competitor_motion',
                'requested': f'all {len(competitor_classes)} competitors are registered '
                             'static objects, so none can carry the differing answer',
                'compiled': 'moving',
                'wanted_by': knob_sources.get('competitor_motion', 'a compiled question'),
            }])
    if not 1 <= result['retry_budget_within_profile'] <= 200:
        raise ValueError('retry_budget_within_profile must be within 1..200')
    if any(not math.isfinite(result[k]) or result[k]<0 for k in ('reserve_tail_s','minimum_overlap_s','min_gap_between_audible_windows_s')):
        raise ValueError('event interval parameters must be finite and nonnegative')
    return result


def neutral_source_declaration(record, actor_id, *, entity_instance_id=None,
                               instance_ordinal=1, source_endpoint_id=None):
    """Declare one physical instance; its identity is the instance, not the asset.

    ``actor_id`` stays the backend slot name (``source1``/``source2``) the UE and
    RLR endpoint conventions are built on, and ``entity_instance_id`` is the
    identity roles, sound events and native facts attach to. Two instances of one
    registered asset therefore stay two entities end to end.
    """
    from avengine.dataset.source_capabilities import make_instance_id
    anchor_id = record['default_emitter_anchor_id']
    anchor = next(a for a in record['emitter_anchors'] if a['anchor_id'] == anchor_id)
    instance_id = str(entity_instance_id or make_instance_id(record['asset_id'], int(instance_ordinal)))
    endpoint = str(source_endpoint_id or f'{actor_id}_mouth')
    binding = {'source_slot_id': actor_id, 'asset_id': record['asset_id'], 'asset_revision': record['revision'],
               'entity_instance_id': instance_id, 'source_endpoint_id': endpoint,
               'semantic_anchor_id': anchor_id, 'emitter_offset_m': deepcopy(anchor['offset_m']),
               'offset_space': anchor['offset_space']}
    if anchor.get('local_basis'):
        binding['local_basis'] = deepcopy(anchor['local_basis'])
    return {'actor_id': actor_id, 'entity_instance_id': instance_id,
            'instance_ordinal': int(instance_ordinal), 'source_endpoint_id': endpoint,
            'asset_id': record['asset_id'], 'asset_revision': record['revision'],
            'entity_class': record['entity_class'], 'identity': deepcopy(record['identity']),
            'realized_attributes': deepcopy(record['realized_attributes']), 'display_label': record['display_label'],
            'emitter_binding': binding, 'timeline': deepcopy(record.get('timeline')),
            'motion_model': 'rigid_static' if record['entity_class'] in RIGID else 'articulated',
            'native_binding_status': 'pending_executor'}


def select_entities(request, profile, registry, rng):
    records = registry['assets']; chosen = []; explicit = request.get('source_asset_ids')
    declared = profile.get('instances') or []
    repeat_policy = profile.get('asset_repeat_policy', 'distinct_assets')
    ordinals = Counter()
    for i, cls in enumerate(profile['source_classes']):
        row = declared[i] if i < len(declared) else {}
        wanted = explicit[i] if explicit is not None else row.get('asset_id')
        used = {x['asset_id'] for x in chosen}
        # Physical instances are never deduplicated by asset: a named asset is
        # taken as often as it is named. Only the free draw prefers a fresh
        # asset, and that preference is a declared policy rather than identity.
        pool = [r for r in records if (r['entity_class'] == cls or cls == 'rigid_static_object' and r['entity_class'] in RIGID)
                and (wanted is None or r['asset_id'] == wanted)
                and (wanted is not None or repeat_policy == 'allow_repeats' or r['asset_id'] not in used)]
        if not pool:
            raise CandidateFailure('assets', 'no_registered_asset_for_'+cls)
        record = pool[int(rng.integers(len(pool)))]
        ordinals[record['asset_id']] += 1
        chosen.append(neutral_source_declaration(
            record, str(row.get('source_slot_id') or f'source{i+1}'),
            entity_instance_id=row.get('entity_instance_id'),
            instance_ordinal=ordinals[record['asset_id']]))
    instance_ids = [actor['entity_instance_id'] for actor in chosen]
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError('two instances of one asset must keep distinct entity_instance_id values')
    endpoints = [actor['source_endpoint_id'] for actor in chosen]
    if len(set(endpoints)) != len(endpoints):
        raise ValueError('scene entities cannot share a source endpoint')
    return chosen


def _gender(value):
    return {'m':'male', 'f':'female', 'male':'male', 'female':'female'}.get(str(value).lower())


LEGACY_SOUND_CLASS_SHIM = 'legacy_sampler_compatibility_shim'


def _legacy_sound_class_config(actor, sound):
    """Express this module's historical matching rules as a declared mapping.

    The decision itself belongs to ``source_capabilities.sound_compatibility``,
    which needs the declared semantic mapping a production request carries under
    ``sound_sources``. When a caller passes none, these three rules reproduce
    exactly what this function used to decide inline, so the existing callers
    keep their behaviour while gender, species and allowlist logic stops being a
    second implementation:

    * a human accepted laughter/cough/sneeze as its non-speech emissions;
    * an animal accepted a class listed by the pool row's own
      ``compatible_sound_classes_by_species`` mapping;
    * a device accepted any class its asset/category allowlist already admitted.

    A request that declares the real mapping never reaches this shim.
    """
    identity = actor.get('identity') or {}
    species = {}
    declared = sound.get('compatible_sound_classes_by_species')
    if isinstance(declared, Mapping):
        species = {str(key): [str(item) for item in value] for key, value in declared.items()}
    objects = {}
    sound_class = str(sound.get('sound_class') or '')
    if actor.get('entity_class') in RIGID and sound_class:
        objects = {str(identity.get('object_type')): [sound_class]}
    return {'species_sound_classes': species, 'object_sound_classes': objects,
            'human_nonverbal_sound_classes': ['laughter', 'cough', 'sneeze']}


def sound_match_verdict(actor, sound, config=None):
    """Decide one (actor, pool sound) pairing with the shared P07 semantics.

    Returns that module's verdict unchanged, so a rejection keeps its state and
    its reason instead of collapsing into False.
    """
    from avengine.dataset.source_capabilities import sound_compatibility
    record = {'asset_id': actor.get('asset_id'), 'entity_class': actor.get('entity_class'),
              'identity': actor.get('identity') or {},
              'realized_attributes': actor.get('realized_attributes') or {}}
    resolved, shim = config, config is None
    if shim:
        resolved = _legacy_sound_class_config(actor, sound)
    verdict = dict(sound_compatibility(record, sound, resolved))
    verdict['decided_by'] = 'avengine.dataset.source_capabilities.sound_compatibility'
    verdict['declared_mapping'] = LEGACY_SOUND_CLASS_SHIM if shim else 'request_declared_mapping'
    return verdict


def sound_matches(actor, sound, config=None):
    """True when the shared semantics admit this pool sound for this actor."""
    return bool(sound_match_verdict(actor, sound, config)['compatible'])


def request_sound_class_config(request):
    """Return the declared semantic sound mapping a request carries, if any."""
    if not isinstance(request, Mapping):
        return None
    selection = request.get('sound_selection')
    sources = [request.get('sound_class_config'), request.get('sound_sources')]
    if isinstance(selection, Mapping):
        sources.append(selection.get('sound_class_config'))
    for value in sources:
        if isinstance(value, Mapping) and value:
            return value
    return None


def clip_budget_samples(profile, clock, config=None):
    """Longest clip this Episode can still place, and where that bound came from.

    ``max_clip_s`` stays honoured when a caller states it. When nobody states
    one, the bound is this Episode's own remaining budget: a prepared segment is
    already an activity-selected crop, so the old 5 s pool filter must not come
    back as an implicit ceiling on it.
    """
    sr = int(clock['sample_rate_hz'])
    deadline = int(clock['sample_count']) - int(round(float(profile['reserve_tail_s']) * sr))
    declared = (config or {}).get('max_clip_s')
    if declared is None:
        return deadline, deadline, 'episode_remaining_budget_after_reserved_tail'
    limit = int(round(float(declared) * sr))
    if limit <= 0:
        raise ValueError('max_clip_s must be positive when it is declared')
    return min(limit, deadline), deadline, 'declared_max_clip_s'


# The order a preallocation key is looked up in. The declared entity instance is
# the primary identity - it is what prepare keys the map by - and the positional
# slot is last, because it is the name most likely to collide with someone
# else's instance id.
PREALLOCATION_KEY_NAMESPACES = ('entity_instance_id', 'actor_id', 'source_slot_id')


def _resolve_preallocation_keys(preallocated, actors):
    """Map each preallocation key onto exactly one actor, or refuse.

    The previous version flattened every name of every actor into one dict, so
    a room where one actor's ``entity_instance_id`` equals another's
    ``source_slot_id`` silently overwrote one of them and handed a speaker the
    other speaker's clips. Keys are resolved per namespace now, the primary
    identity wins, and anything genuinely ambiguous is refused by name rather
    than decided by dictionary order.
    """
    namespaces = {field: {} for field in PREALLOCATION_KEY_NAMESPACES}
    for index, actor in enumerate(actors):
        for field in PREALLOCATION_KEY_NAMESPACES:
            name = actor.get(field)
            if not name:
                continue
            namespaces[field].setdefault(str(name), []).append(index)

    resolved, unknown, ambiguous = {}, [], []
    for key, value in preallocated.items():
        text = str(key)
        matches = None
        for field in PREALLOCATION_KEY_NAMESPACES:
            found = namespaces[field].get(text)
            if not found:
                continue
            if len(set(found)) > 1:
                # Two actors answer to this name inside one namespace.
                ambiguous.append({'key': text, 'namespace': field,
                                  'actors': [actors[i].get('actor_id') for i in found]})
                matches = None
                break
            matches = found[0]
            break
        if matches is None:
            if not any(entry['key'] == text for entry in ambiguous):
                unknown.append(text)
            continue
        if matches in resolved:
            ambiguous.append({'key': text, 'namespace': 'collision',
                              'actors': [actors[matches].get('actor_id')],
                              'detail': 'another preallocation key already claimed this actor'})
            continue
        resolved[matches] = value

    if unknown:
        known = sorted({str(actor.get(field)) for actor in actors
                        for field in PREALLOCATION_KEY_NAMESPACES if actor.get(field)})
        raise ValueError(
            'preallocated sounds contain an unknown actor: ' + ', '.join(sorted(unknown))
            + '; this request declares ' + ', '.join(known))
    if ambiguous:
        raise ValueError(
            'preallocated sounds name an actor ambiguously and would overwrite one '
            'another: ' + '; '.join(
                f"{row['key']!r} via {row['namespace']} -> {row['actors']}"
                for row in ambiguous))
    return {actors[index]['actor_id']: value for index, value in resolved.items()}


def select_sounds(actors, sounds, profile, clock, request, rng):
    sr = int(clock['sample_rate_hz']); config = request.get('sound_selection', {})
    clip_policy = str(config.get('clip_span_fit_policy', CLIP_SPAN_FIT_POLICY) or CLIP_SPAN_FIT_POLICY)
    if clip_policy != CLIP_SPAN_FIT_POLICY:
        raise ValueError('unsupported clip_span_fit_policy: ' + clip_policy)
    sound_class_config = request_sound_class_config(request)
    max_samples, deadline, clip_bound_source = clip_budget_samples(profile, clock, config)
    speakers = profile['speaking_indices']; order = list(speakers); rng.shuffle(order)
    preallocated = config.get('preallocated_sound_asset_ids_by_actor')
    if preallocated is not None:
        if not isinstance(preallocated, Mapping):
            raise ValueError('preallocated sounds must be an actor-to-sound-ID mapping')
        # prepare keys this map by the instance the request declared, while the
        # actor carries the slot name (source1..sourceN). Reading only the slot
        # meant every request whose instances had descriptive names died here
        # with "unknown actor", so both names of the same entity are accepted.
        preallocated = _resolve_preallocation_keys(preallocated, actors)
        for i in speakers:
            allowed = preallocated.get(actors[i]['actor_id'])
            if not isinstance(allowed, list) or any(not isinstance(value, str) or not value for value in allowed):
                raise ValueError('preallocated sounds must explicitly cover every speaking actor')
    pools = {}
    for i in speakers:
        allowed = None if preallocated is None else preallocated[actors[i]['actor_id']]
        # The prepared clip's own length is the bound; the length of the original
        # recording it was cropped from is not a rejection reason.
        pool = [s for s in sounds if (allowed is None or s.get('sound_asset_id') in allowed)
                and sound_matches(actors[i], s, sound_class_config) and 0 < int(s['sample_count']) <= max_samples
                and int(s.get('sample_rate_hz', sr)) == sr
                and (s.get('sound_class') not in {'speech','speech_playback'} or float(s.get('active_duration_s', 0)) >= float(config.get('min_audible_s', 1.5)))]
        if not pool:
            raise CandidateFailure('sounds', 'no_compatible_prepared_sound_'+actors[i]['actor_id'])
        pools[i] = pool
    repeated = None
    if profile['event_relation']=='repeat':
        requested_repeat = config.get('repeat_actor_id')
        if requested_repeat:
            matching = [i for i in speakers if actors[i]['actor_id'] == requested_repeat]
            if len(matching) != 1:
                raise ValueError('repeat_actor_id must name exactly one speaking actor')
            repeated = matching[0]
        else:
            repeated = speakers[int(rng.integers(len(speakers)))]
    selected = {}; transcripts = set(); remaining = deadline
    gap = int(round(profile['min_gap_between_audible_windows_s'] * sr))
    for pos, i in enumerate(order):
        rest = order[pos+1:]
        # Retain a feasible duration suffix before drawing a clip.
        suffix = sum(min(int(s['sample_count']) for s in pools[j])*(2 if j==repeated else 1) for j in rest) + gap * (len(rest)+int(repeated in rest))
        limit = (remaining-suffix-gap*int(i==repeated))//(2 if i==repeated else 1) if profile['event_relation'] in {'sequential','repeat'} else deadline
        pool = [s for s in pools[i] if int(s['sample_count']) <= limit and
                (not config.get('unique_first_utterance_transcripts', True) or not s.get('transcript') or s['transcript'] not in transcripts)]
        if not pool:
            raise CandidateFailure('sounds', 'clip_budget_or_transcript_candidates_exhausted')
        sound = deepcopy(pool[int(rng.integers(len(pool)))]); sound['actor_id'] = actors[i]['actor_id']
        # The emission belongs to a physical instance and an endpoint, so two
        # instances of one asset never merge into one source downstream.
        sound['entity_instance_id'] = actors[i]['entity_instance_id']
        sound['source_endpoint_id'] = actors[i]['source_endpoint_id']
        sound['clip_length_bound_source'] = clip_bound_source
        sound['clip_length_bound_samples'] = int(max_samples)
        if i==repeated:sound['repeat_requested']=True
        selected[i] = sound
        if sound.get('transcript'): transcripts.add(sound['transcript'])
        remaining -= (int(sound['sample_count']) + gap)*(2 if i==repeated else 1)
    return selected


def _moving_flags(profile, actors, rng, motion_requirements=None):
    """Decide who moves. Solver requirements replace random actor flags.

    ``speech_motion`` is the statement about the anchor, and
    ``competitor_motion`` is the separate statement about everyone else. Under
    the historical default ``any`` the competitors still receive the old random
    flag; ``still`` and ``moving`` make target and competitor answer the same
    motion question differently, which is what an answerable QA-06/QA-15/QA-17
    item needs.

    ``moving`` reads two ways on purpose. Stated by the caller it means every
    competitor, and a registered static object among them is a contradiction.
    Compiled from a question it means at least one competitor answers
    differently, matching ``conditioned_motion._competitor_duty`` and the gate in
    ``unified_catalog._p8_apply_distractor_gate``; an immobile device is then
    left still rather than voiding a legal human plus animal plus device group.
    """
    if motion_requirements is not None:
        flags = []
        for actor in actors:
            requirement = motion_requirements.get(actor.get('entity_instance_id'))
            if requirement is None:
                flags.append(False)
                continue
            if requirement.must_move and requirement.moving_frames is None:
                raise CandidateFailure(
                    'routes', 'motion_solver_required_motion_window_missing')
            flags.append(bool(requirement.moving_frames))
        return flags
    flags = [False]*len(actors)
    anchors = set(profile['anchor_indices'])
    competitors = [i for i in range(len(actors)) if i not in anchors]
    articulated = [i for i in competitors if actors[i]['entity_class'] not in RIGID]
    competitor_motion = profile.get('competitor_motion', COMPETITOR_MOTION_DEFAULT)
    if profile['speech_motion'] == 'speaker_moving':
        for i in profile['anchor_indices']: flags[i] = True
    elif profile['speech_motion'] == 'competitor_moving':
        if not articulated:
            raise CandidateFailure('routes', 'no_articulated_competitor_for_moving_profile')
        if competitor_motion == 'moving':
            for i in articulated: flags[i] = True
        else:
            flags[articulated[int(rng.integers(len(articulated)))]] = True
    if competitor_motion == 'moving':
        if not competitors:
            raise CandidateFailure('routes', 'no_competitor_for_required_competitor_motion')
        # A caller who states the knob means every competitor. A compiled question
        # means "at least one competitor answers differently", which is what
        # unified_catalog._p8_apply_distractor_gate actually reads, so a device
        # that cannot walk is left still instead of voiding the whole group.
        stated_by_caller = (profile.get('knob_sources') or {}).get(
            'competitor_motion') == 'request_profile'
        if stated_by_caller and len(articulated) != len(competitors):
            raise CandidateFailure('routes', 'static_entity_cannot_satisfy_required_motion')
        if not articulated:
            raise CandidateFailure('routes', 'no_locomotion_capable_competitor')
        for i in articulated: flags[i] = True
    elif competitor_motion == 'still':
        for i in competitors: flags[i] = False
    elif profile['speech_motion'] == 'speaker_moving':
        for i in articulated: flags[i] = bool(rng.integers(2))
    if any(flags[i] and a['entity_class'] in RIGID for i,a in enumerate(actors)):
        raise CandidateFailure('routes', 'static_entity_cannot_satisfy_required_motion')
    return flags



def _motion_budget(request, profile):
    from avengine.rooms.conditioned_motion import MotionBudget
    values = dict(profile)
    motion = request.get('motion') if isinstance(request.get('motion'), Mapping) else {}
    requested_profile = (
        request.get('profile') if isinstance(request.get('profile'), Mapping) else {}
    )
    for name in (
        'reserve_tail_s', 'walk_speed_range_mps', 'moving_threshold_mps',
        'min_gap_between_audible_windows_s', 'earliest_start_s', 'end_hold_s',
        'anchor_pre_silence_s', 'max_clip_s', 'minimum_motion_s',
        'public_time_precision', 'moving_flag_convention',
    ):
        if name in values:
            continue
        value = motion.get(name) if isinstance(motion, Mapping) else None
        if value is None:
            value = requested_profile.get(name)
        if value is None:
            value = request.get(name)
        if value is not None:
            values[name] = deepcopy(value)
    return MotionBudget.from_profile(values)


def _constructive_motion_requested(request, profile):
    """Read the opt-in route-construction mode without changing old callers.

    The motion solver is still used to derive the legal event/query window and
    per-instance requirements.  This flag only selects how an already-solved
    requirement is consumed by ``sample_routes``; absent means the historical
    random route sampler remains in charge.
    """
    raw = profile.get('constructive_motion', False)
    if isinstance(raw, bool):
        return raw
    raise ValueError('constructive_motion must be a boolean')


def _schedule_for_motion_solver(sounds, profile, clock, rng):
    events = _event_bindings(sounds, profile, rng)
    all_frames = np.ones(int(clock['frame_count']), dtype=bool)
    starts = {
        key: legal_start_ranges(all_frames, event, clock, profile)
        for key, event in events.items()
    }
    if any(not values for values in starts.values()):
        raise CandidateFailure('schedule', 'no_legal_event_start_for_motion_solver')
    schedule = schedule_legal_events(events, starts, clock, profile, rng)
    if not isinstance(schedule, Mapping):
        raise CandidateFailure('schedule', 'no_legal_event_schedule_for_motion_solver')
    sr = int(clock['sample_rate_hz'])
    event_start_s = {}
    other_windows_s = []
    for key in sorted(events):
        event = events[key]
        start = int(schedule[key])
        entity = event.get('entity_instance_id')
        if isinstance(entity, str) and entity:
            event_start_s.setdefault(entity, start / sr)
        other_windows_s.append((start / sr, (start + int(event['sample_count'])) / sr))
    return events, dict(schedule), event_start_s, tuple(other_windows_s)


def _compiled_target_instance(compiled):
    return next(
        (str(item.entity_instance_id) for item in getattr(compiled, 'subjects', ())
         if getattr(item, 'role', None) == 'target'),
        None,
    )


def _merge_motion_requirement(existing, incoming):
    from dataclasses import replace
    if (
        existing.moving_frames is not None
        and incoming.moving_frames is not None
        and tuple(existing.moving_frames) != tuple(incoming.moving_frames)
    ):
        raise CandidateFailure(
            'question', 'conflicting_motion_windows_for_same_instance')
    moving = existing.moving_frames or incoming.moving_frames
    still = tuple(sorted(set(
        tuple(int(value) for value in window)
        for window in (*existing.still_frames, *incoming.still_frames)
    )))
    if moving is not None:
        first, last = map(int, moving)
        if any(max(first, a) < min(last, b) for a, b in still):
            raise CandidateFailure(
                'question', 'motion_and_still_windows_overlap_for_same_instance')
    base = (
        incoming
        if existing.semantics == 'unconstrained' and incoming.semantics != 'unconstrained'
        else existing
    )
    return replace(
        base,
        must_move=bool(existing.must_move or incoming.must_move),
        moving_frames=moving,
        permitted_moving_frames=(
            existing.permitted_moving_frames or incoming.permitted_moving_frames
        ),
        still_frames=still,
        required_path_length_range_m=(
            existing.required_path_length_range_m
            or incoming.required_path_length_range_m
        ),
        required_moving_steps=(
            existing.required_moving_steps or incoming.required_moving_steps
        ),
    )


def _solve_motion_conditions(
    request, profile, source_registry, actors, selected_sounds, compiled_conditions,
    clock, event_start_s, other_event_windows_s,
):
    from avengine.qa import generation_conditions as gc
    from avengine.rooms.conditioned_motion import solve_motion_windows
    available = [
        item for item in compiled_conditions
        if getattr(item, 'state', None) == gc.STATE_AVAILABLE
    ]
    if not available:
        return None
    anchor_ids = {
        str(actors[index]['entity_instance_id'])
        for index in profile['anchor_indices']
    }
    selected = [
        item for item in available
        if _compiled_target_instance(item) in anchor_ids
    ] or available
    sounds_by_instance = {
        str(value['entity_instance_id']): value
        for value in selected_sounds.values()
        if isinstance(value, Mapping) and value.get('entity_instance_id')
    }
    budget = _motion_budget(request, profile)
    solutions = []
    for compiled in selected:
        solution = solve_motion_windows(
            compiled,
            clock=clock,
            budget=budget,
            sounds=sounds_by_instance,
            registry=source_registry,
            camera={'motion': 'static'},
            event_start_s=event_start_s,
            other_event_windows_s=other_event_windows_s,
        )
        if not solution.solved:
            codes = ','.join(str(row['code']) for row in solution.rejections)
            raise CandidateFailure(
                'question', 'motion_solver_rejected' + (':' + codes if codes else ''))
        solutions.append(solution)
    requirements = {}
    for solution in solutions:
        for requirement in solution.requirements:
            existing = requirements.get(requirement.entity_instance_id)
            requirements[requirement.entity_instance_id] = (
                requirement if existing is None
                else _merge_motion_requirement(existing, requirement)
            )
    primary = next(
        (solution for solution in solutions if solution.distance_trend),
        solutions[0],
    )
    concrete_requirements = tuple(requirements.values())
    if not any(
        requirement.must_move
        or requirement.moving_frames is not None
        or requirement.still_frames
        or requirement.required_path_length_range_m is not None
        or requirement.required_moving_steps is not None
        for requirement in concrete_requirements
    ):
        # A visibility-only recipe still needs the caller's ordinary motion
        # profile to decide whether a real route moves. An unconstrained
        # MotionSolution must not turn speaker_moving into all-still merely
        # because the compiler emitted an event for the visibility question.
        return None
    return {
        'budget': budget,
        'solutions': tuple(solutions),
        'primary': primary,
        'requirements': requirements,
        'event_start_s': dict(event_start_s),
        'other_event_windows_s': tuple(other_event_windows_s),
    }


def _motion_context_record(context):
    if context is None:
        return None
    return {
        'status': 'solved',
        'event_schedule_source': 'conditioned_sampler.schedule_legal_events',
        'event_start_s': dict(context['event_start_s']),
        'other_event_windows_s': [
            list(window) for window in context['other_event_windows_s']
        ],
        'budget': context['budget'].to_dict(),
        'solutions': [solution.to_dict() for solution in context['solutions']],
        'requirements': [
            context['requirements'][key].to_dict()
            for key in sorted(context['requirements'])
        ],
    }


PINNED_ENDPOINT_TOLERANCE_M = 1e-5


def _resolve_pinned_positions(pinned, flags):
    """Validate the caller's pins against the moving flags, keyed by actor index."""
    if pinned is None:
        return {}
    if not isinstance(pinned, Sequence) or isinstance(pinned, (str, bytes)):
        raise ValueError('pinned_static_positions_m must be a sequence aligned to the actors')
    if len(pinned) != len(flags):
        raise ValueError(
            'pinned_static_positions_m must have one entry per actor, using null for '
            'an actor whose position is not pinned')
    resolved = {}
    for index, value in enumerate(pinned):
        if value is None:
            continue
        point = np.asarray(value, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError('each pinned static position must be three finite metres')
        if flags[index]:
            # A pin fixes where a body stands for the whole clip. Asking the same
            # body to walk is a contradiction, and silently honouring one of the
            # two would produce a plan that answers neither statement.
            raise CandidateFailure(
                'routes', 'pinned_static_position_requested_for_a_moving_actor')
        resolved[index] = point
    return resolved


def _endpoint_matching(points, pinned):
    """Return the route endpoint that equals ``pinned``, or None."""
    for endpoint in (points[0], points[-1]):
        if float(np.linalg.norm(np.asarray(endpoint, dtype=float) - pinned)) <= (
                PINNED_ENDPOINT_TOLERANCE_M):
            return np.asarray(endpoint, dtype=float)
    return None


def _groups_matching_pins(groups, bank, pins):
    """Keep the groups that can seat every pinned actor on a route of its own."""
    endpoints = {}

    def route_endpoints(route_index):
        if route_index not in endpoints:
            pts = np.asarray(bank[int(route_index)]['points_m'], dtype=float)
            endpoints[route_index] = (pts[0], pts[-1])
        return endpoints[route_index]

    kept, assignments = [], []
    for group in groups:
        options = {}
        for actor_index, point in pins.items():
            options[actor_index] = [
                route for route in group
                if any(float(np.linalg.norm(np.asarray(end, dtype=float) - point))
                       <= PINNED_ENDPOINT_TOLERANCE_M
                       for end in route_endpoints(route))
            ]
            if not options[actor_index]:
                break
        else:
            seating = _seat_pins(sorted(options), options, {})
            if seating is not None:
                kept.append(group)
                assignments.append(seating)
    return kept, assignments


def _seat_pins(order, options, taken):
    """Give each pinned actor a distinct route, or report that none fits."""
    if not order:
        return dict(taken)
    actor_index, rest = order[0], order[1:]
    for route in options[actor_index]:
        if route in taken.values():
            continue
        seated = _seat_pins(rest, options, {**taken, actor_index: route})
        if seated is not None:
            return seated
    return None


def _order_ids_for_pins(ids, seating, flags, rng):
    """Place each route at its actor's index, shuffling only the unpinned ones."""
    ordered = [None] * len(flags)
    for actor_index, route in seating.items():
        ordered[actor_index] = route
    free = [route for route in ids if route not in seating.values()]
    rng.shuffle(free)
    for actor_index in range(len(ordered)):
        if ordered[actor_index] is None:
            ordered[actor_index] = free.pop()
    return ordered


def _order_ids_for_motion(ids, seating, flags, requirements, bank, rng):
    """Assign long-enough retained routes to the required moving actors first."""
    ordered = [None] * len(flags)
    used = set()
    for actor_index, route in (seating or {}).items():
        ordered[int(actor_index)] = int(route)
        used.add(int(route))
    free = [int(route) for route in ids if int(route) not in used]
    moving = [
        index for index, flag in enumerate(flags)
        if flag and getattr(requirements.get(index), 'moving_frames', None) is not None
    ]
    for actor_index in moving:
        requirement = requirements[actor_index]
        required_range = getattr(requirement, 'required_path_length_range_m', None)
        minimum = float(required_range[0]) if required_range else 0.0
        choices = sorted(
            (route for route in free
             if _polyline_length(bank[int(route)]['points_m']) + 1.0e-9 >= minimum),
            key=lambda route: (
                -_polyline_length(bank[int(route)]['points_m']),
                str(bank[int(route)].get('route_id', route)),
            ),
        )
        if not choices:
            raise CandidateFailure('routes', 'motion_solver_required_path_length_unavailable')
        chosen = choices[0]
        ordered[actor_index] = chosen
        free.remove(chosen)
    rng.shuffle(free)
    for actor_index in range(len(ordered)):
        if ordered[actor_index] is None:
            ordered[actor_index] = free.pop()
    return ordered


def _native_routes(space, flags, frames, fps, rng, *, start_hold_frames=None,
                   pinned_static_positions_m=None,
                   constructive_motion_requirements=None):
    """Draw one legal native route group, optionally pinning static start points.

    ``pinned_static_positions_m`` is a list aligned to ``flags``: an entry is a
    3-vector for an actor whose static position the caller fixes, or ``None`` to
    leave that actor alone. The caller resolves instance identity into this
    order, so the pin follows the instance rather than a route index.

    A pin is matched against the real endpoints of the retained route bank, so
    it selects among routes that already exist instead of inventing a position.
    Every pinned actor has to land on an endpoint of its own route, and two
    pinned actors cannot share one route. Nothing is pinned by default and the
    old random endpoint draw is kept for every unpinned actor.
    """
    from avengine.rooms.native_qa_room import _hold_endpoint_path
    if float(space.frame_rate_hz) != fps:
        raise CandidateFailure('routes', 'native_route_clock_mismatch')
    bank = space.route_bank()
    if len(bank) < len(flags):raise CandidateFailure('routes','insufficient_native_routes')
    # Reuse the prior audit's complete native clique enumeration, now in production.
    # It keeps every unchanged native pair constraint and has no scored prefix.
    cache=getattr(space,'_qa_native_group_cache',{})
    if frames not in cache:
        full=np.asarray([_hold_endpoint_path(np.asarray(row['points_m'],dtype=float),frames) for row in bank])
        adjacency=[set() for _ in bank];pairs=[];triples=[];quads=[]
        for a in range(len(full)):
            candidates=np.arange(a+1,len(full));first=np.linalg.norm(full[candidates,0]-full[a,0],axis=1);last=np.linalg.norm(full[candidates,-1]-full[a,-1],axis=1)
            candidates=candidates[(first>=.95)&(last>=.95)&(first<=3.5)&(last<=3.5)]
            if len(candidates):
                good=np.all(np.sum((full[candidates]-full[a])**2,axis=-1)>=.95**2,axis=-1)
                for b in candidates[good]:
                    b=int(b);adjacency[a].add(b);adjacency[b].add(a);pairs.append((a,b))
        for a,b in pairs:
            common=sorted(c for c in adjacency[a]&adjacency[b] if c>b)
            for c in common:
                triples.append((a,b,c))
                for d in sorted(adjacency[c].intersection(common)):
                    if d>c:quads.append((a,b,c,d))
        cache[frames]={2:pairs,3:triples,4:quads};setattr(space,'_qa_native_group_cache',cache)
    groups=cache[frames][len(flags)]
    if not groups:raise CandidateFailure('routes','no_legal_native_route_group')
    pins = _resolve_pinned_positions(pinned_static_positions_m, flags)
    assignment = None
    if pins:
        groups, assignment = _groups_matching_pins(groups, bank, pins)
        if not groups:
            raise CandidateFailure(
                'routes', 'no_legal_native_route_group_at_pinned_static_positions')
    if constructive_motion_requirements:
        moving = [
            index for index, flag in enumerate(flags)
            if flag and getattr(
                constructive_motion_requirements.get(index), 'moving_frames', None
            ) is not None
        ]
        if moving:
            if assignment is None:
                groups = [
                    group for group in groups
                    if sum(
                        _polyline_length(bank[int(route)]['points_m']) + 1.0e-9
                        >= float(
                            getattr(
                                constructive_motion_requirements[index],
                                'required_path_length_range_m',
                                (0.0, 0.0),
                            )[0]
                        )
                        for route in group
                    ) >= len(moving)
                ]
            else:
                kept_groups, kept_assignments = [], []
                for group, seating in zip(groups, assignment):
                    if all(
                        _polyline_length(bank[int(seating[index])]['points_m']) + 1.0e-9
                        >= float(
                            getattr(
                                constructive_motion_requirements[index],
                                'required_path_length_range_m',
                                (0.0, 0.0),
                            )[0]
                        )
                        for index in moving
                        if index in seating
                    ):
                        kept_groups.append(group)
                        kept_assignments.append(seating)
                groups, assignment = kept_groups, kept_assignments
            if not groups:
                raise CandidateFailure(
                    'routes', 'motion_solver_required_path_length_unavailable')
    choice = int(rng.integers(len(groups)))
    ids = list(groups[choice])
    if constructive_motion_requirements:
        seating = None if assignment is None else assignment[choice]
        requirements_by_index = {
            index: constructive_motion_requirements.get(index)
            for index in range(len(flags))
        }
        ids = _order_ids_for_motion(
            ids, seating, flags, requirements_by_index, bank, rng)
    elif assignment is None:
        rng.shuffle(ids)
    else:
        # The pinned actors already name which route each of them takes, so the
        # order is fixed by the pins and only the free actors are shuffled.
        ids = _order_ids_for_pins(ids, assignment[choice], flags, rng)
    paths = []
    for i, selected in enumerate(ids):
        p = np.asarray(bank[int(selected)]['points_m'], dtype=float)
        if flags[i]:
            # Delay is legal only when the full captured route clock is kept.
            max_delay = max(0, frames-len(p))
            if start_hold_frames is not None and (isinstance(start_hold_frames,bool) or not isinstance(start_hold_frames,int) or not 0<=start_hold_frames<=max_delay):
                raise CandidateFailure('routes','requested_native_delay_cannot_preserve_full_route')
            delay = int(rng.integers(max_delay+1)) if start_hold_frames is None else start_hold_frames
            path = _hold_endpoint_path(p, frames, start_hold_frames=delay)
        else:
            pinned = None if not pins else pins.get(i)
            if pinned is None:
                endpoint = p[int(rng.integers(2)) * (len(p)-1)]
            else:
                endpoint = _endpoint_matching(p, pinned)
                if endpoint is None:
                    raise CandidateFailure(
                        'routes', 'pinned_static_position_is_not_an_endpoint_of_its_route')
            path = np.repeat(endpoint[None], frames, axis=0)
        paths.append(path)
    if any(np.linalg.norm(a-b, axis=1).min() < .95 for a,b in itertools.combinations(paths, 2)):
        raise CandidateFailure('routes', 'native_group_separation_below_0.95_m')
    if any(np.linalg.norm(a[0]-b[0]) > 3.5 or np.linalg.norm(a[-1]-b[-1]) > 3.5 for a,b in itertools.combinations(paths,2)):
        raise CandidateFailure('routes', 'native_group_endpoint_separation_above_3.5_m')
    return paths, {
        'authority': space.metadata.get('route_authority', 'retained_native_route_bank'),
        'selected_route_ids': [bank[int(i)]['route_id'] for i in ids],
        'selected_route_points_m': [
            np.asarray(bank[int(i)]['points_m'], dtype=float).tolist() for i in ids
        ],
        'selection': ('uniform_over_legal_native_groups_at_pinned_static_positions'
                      if pins else 'uniform_over_all_legal_native_groups'),
        'legal_native_group_count': len(groups),
        'minimum_separation_m': .95,
        'pinned_static_actor_indices': sorted(pins) if pins else [],
        'pinned_endpoint_tolerance_m': PINNED_ENDPOINT_TOLERANCE_M if pins else None,
    }



def _polyline_length(polyline):
    points = np.asarray(polyline, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _polyline_prefix(polyline, length_m):
    points = np.asarray(polyline, dtype=float)
    target = float(length_m)
    result = [points[0]]
    remaining = target
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        segment = float(np.linalg.norm(delta))
        if segment <= 1.0e-12:
            continue
        if remaining >= segment - 1.0e-9:
            result.append(end)
            remaining -= segment
            continue
        result.append(start + delta * (remaining / segment))
        break
    if len(result) < 2:
        result.append(points[-1])
    return np.asarray(result, dtype=float)


def _fit_motion_polyline(polyline, required_range):
    if required_range is None:
        return np.asarray(polyline, dtype=float)
    low, high = map(float, required_range)
    available = _polyline_length(polyline)
    if available + 1.0e-9 < low:
        return None
    target = min(available, max(low, (low + high) / 2.0))
    return _polyline_prefix(polyline, target)


def _apply_motion_requirements(
    space, actors, paths, metadata, clock, motion_requirements, motion_budget,
):
    from avengine.rooms.conditioned_motion import EpisodeClock, build_motion_trajectory
    solver_clock = (
        clock if isinstance(clock, EpisodeClock)
        else EpisodeClock.from_mapping(clock)
    )
    actor_records = metadata.setdefault('actors', {})
    native_polylines = metadata.get('selected_route_points_m') or ()
    for index, actor in enumerate(actors):
        requirement = motion_requirements.get(actor.get('entity_instance_id'))
        if requirement is None or requirement.moving_frames is None:
            continue
        if native_polylines:
            polyline = native_polylines[index]
            record = actor_records.setdefault(actor['actor_id'], {})
        elif isinstance(actor_records, Mapping):
            record = actor_records.setdefault(actor['actor_id'], {})
            polyline = record.get('polyline_habitat_m') or record.get('route_points_m')
        elif isinstance(actor_records, list):
            record = actor_records[index] if index < len(actor_records) else {}
            polyline = (
                (record.get('polyline_habitat_m') or record.get('route_points_m'))
                if isinstance(record, Mapping) else None
            )
        else:
            record = {}
            polyline = None
        if polyline is None or _polyline_length(polyline) <= 0.0:
            raise CandidateFailure(
                'routes', 'motion_solver_navigation_polyline_unavailable')
        selected_polyline = _fit_motion_polyline(
            polyline, requirement.required_path_length_range_m)
        if selected_polyline is None:
            raise CandidateFailure(
                'routes', 'motion_solver_required_path_length_unavailable')
        built = build_motion_trajectory(
            polyline_m=selected_polyline,
            moving_frames=requirement.moving_frames,
            clock=solver_clock,
            budget=motion_budget,
        )
        if not built['speed_within_declared_range']:
            raise CandidateFailure(
                'routes', 'motion_solver_speed_outside_declared_range')
        path = np.asarray(built['path_m'], dtype=float)
        if not all(space.is_navigable(point) for point in path):
            raise CandidateFailure(
                'routes', 'motion_solver_path_left_existing_navigation')
        paths[index] = path
        if not isinstance(record, dict):
            record = dict(record)
            if isinstance(actor_records, list) and index < len(actor_records):
                actor_records[index] = record
            elif isinstance(actor_records, Mapping):
                actor_records[actor['actor_id']] = record
        solver_record = dict(built)
        solver_record['path_m'] = np.asarray(built['path_m']).tolist()
        solver_record['moving_flags'] = np.asarray(
            built['moving_flags'], dtype=bool).tolist()
        record.update({
            'motion': 'solver_conditioned_walk',
            'motion_construction': metadata.get(
                'motion_construction', 'legacy_random_route_then_solver'),
            'motion_window_frames': list(requirement.moving_frames),
            'required_path_length_range_m': (
                None if requirement.required_path_length_range_m is None
                else list(requirement.required_path_length_range_m)
            ),
            'motion_solver': solver_record,
            'polyline_habitat_m': selected_polyline.tolist(),
            'path_length_m': float(built['path_length_m']),
            'speed_max_mps': float(built['maximum_step_speed_mps']),
            'all_sampled_centers_navigable': True,
        })
    if any(
        np.linalg.norm(np.asarray(left) - np.asarray(right), axis=1).min() < .95
        for left, right in itertools.combinations(paths, 2)
    ):
        raise CandidateFailure(
            'routes', 'motion_solver_path_separation_below_0.95_m')


def _constructive_candidate_points(space, region, rng):
    """Read a bounded deterministic candidate set from existing navigation."""
    points_reader = getattr(space, 'points', None)
    if callable(points_reader):
        raw = points_reader(region)
        source = 'space.points_lexicographic'
    else:
        # Habitat's adapter exposes seeded point queries rather than a raster
        # point list. A bounded seeded pool keeps this mode from redrawing a
        # complete world while retaining the adapter's navigation authority.
        candidate_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        raw = []
        for _ in range(96):
            try:
                point = space.sample_navigable(candidate_rng, region)
            except (ValueError, RuntimeError):
                continue
            raw.append(point)
        source = 'bounded_seeded_sample_navigable'
    points = np.asarray(raw, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise CandidateFailure('routes', 'constructive_navigation_candidates_invalid')
    unique = {}
    for point in points:
        if not np.all(np.isfinite(point)):
            continue
        if region is not None and not np.all(
            (point >= np.asarray(region)[0]) & (point <= np.asarray(region)[1])
        ):
            continue
        if not space.is_navigable(point):
            continue
        unique.setdefault(tuple(np.round(point, 6)), np.asarray(point, dtype=float))
    ordered = [unique[key] for key in sorted(unique)]
    if not ordered:
        raise CandidateFailure('routes', 'constructive_navigation_candidates_empty')
    return np.asarray(ordered, dtype=float), source


def _constructive_motion_paths(
    space, actors, profile, clock, rng, region, room, motion_requirements,
    motion_budget,
):
    """Construct solver windows from legal points before camera selection."""
    from avengine.rooms.conditioned_motion import EpisodeClock, build_motion_trajectory

    frames = int(clock['frame_count'])
    solver_clock = (
        clock if isinstance(clock, EpisodeClock)
        else EpisodeClock.from_mapping(clock)
    )
    floor_region, floor_y = lock_same_floor_region(space, rng, region, room)
    candidates, candidate_source = _constructive_candidate_points(
        space, floor_region, rng)
    requirements = {
        index: motion_requirements.get(actor.get('entity_instance_id'))
        for index, actor in enumerate(actors)
    }
    moving_indices = [
        index for index, requirement in requirements.items()
        if requirement is not None and requirement.moving_frames is not None
    ]
    if len(candidates) < len(actors):
        raise CandidateFailure(
            'routes', 'constructive_navigation_candidates_insufficient')

    center = (
        candidates[:, [0, 2]].min(axis=0)
        + candidates[:, [0, 2]].max(axis=0)
    ) / 2.0
    ordered_candidates = sorted(
        candidates,
        key=lambda point: (
            float(np.linalg.norm(point[[0, 2]] - center)),
            tuple(float(value) for value in point),
        ),
    )
    starts = []
    for index in range(len(actors)):
        eligible = [
            point for point in candidates
            if all(np.linalg.norm(point - prior) >= 0.95 for prior in starts)
        ]
        if index == 0:
            selected = next(
                (
                    point for point in ordered_candidates
                    if any(np.array_equal(point, item) for item in eligible)
                ),
                None,
            )
        else:
            selected = max(
                eligible,
                key=lambda point: min(
                    np.linalg.norm(point - prior) for prior in starts
                ),
                default=None,
            )
        if selected is None:
            raise CandidateFailure(
                'routes', 'constructive_initial_source_separation_unavailable')
        starts.append(np.asarray(selected, dtype=float))

    paths = [
        np.repeat(start[None], frames, axis=0)
        for start in starts
    ]
    records = [
        {'motion': 'static', 'route_points_m': None}
        for _ in actors
    ]
    for index in moving_indices:
        requirement = requirements[index]
        start = starts[index]
        selected = None
        required_range = requirement.required_path_length_range_m
        desired_length = (
            float(sum(required_range) / 2.0)
            if required_range is not None else 0.0
        )
        endpoint_candidates = sorted(
            candidates,
            key=lambda point: (
                abs(float(np.linalg.norm(point - start)) - desired_length),
                tuple(float(value) for value in point),
            ),
        )
        for end in endpoint_candidates:
            if np.linalg.norm(end - start) < 1.0e-8:
                continue
            if any(
                actor_index != index
                and np.linalg.norm(end - starts[actor_index]) < 0.95
                for actor_index in range(len(starts))
            ):
                continue
            polyline = space.shortest_path(start, end)
            if polyline is None:
                continue
            polyline = np.asarray(polyline, dtype=float)
            if polyline.ndim != 2 or polyline.shape[0] < 2 or polyline.shape[1] != 3:
                continue
            fitted = _fit_motion_polyline(polyline, required_range)
            if fitted is None:
                continue
            try:
                built = build_motion_trajectory(
                    polyline_m=fitted,
                    moving_frames=requirement.moving_frames,
                    clock=solver_clock,
                    budget=motion_budget,
                )
            except (TypeError, ValueError):
                continue
            if not built['speed_within_declared_range']:
                continue
            path = np.asarray(built['path_m'], dtype=float)
            if not all(space.is_navigable(point) for point in path):
                continue
            if any(
                np.linalg.norm(path - other, axis=1).min() < 0.95
                for actor_index, other in enumerate(paths)
                if actor_index != index
            ):
                continue
            if any(
                np.linalg.norm(path - starts[actor_index], axis=1).min() < 0.95
                for actor_index in range(index + 1, len(starts))
            ):
                continue
            selected = (polyline, built)
            break
        if selected is None:
            raise CandidateFailure(
                'routes', 'motion_solver_required_path_length_unavailable')
        polyline, built = selected
        paths[index] = np.asarray(built['path_m'], dtype=float)
        records[index] = {
            'motion': 'solver_conditioned_walk',
            'route_points_m': np.asarray(polyline, dtype=float).tolist(),
            'required_contiguous_motion_frames': list(requirement.moving_frames),
        }

    metadata = {
        'authority': getattr(space, 'metadata', {}).get(
            'authority', 'existing_navigation'),
        'actors': records,
        'minimum_separation_m': 0.95,
        'selected_floor_height_m': floor_y,
        'same_floor_tolerance_m': SAME_FLOOR_Y_TOLERANCE_M,
        'motion_construction': 'constructive_existing_navigation',
        'motion_candidate_source': candidate_source,
        'motion_candidate_count': int(len(candidates)),
    }
    return paths, metadata, floor_y, 'constructive_existing_navigation'


def _static_placement_rows(placement_plan):
    if not isinstance(placement_plan, Mapping):
        return {}
    rows = placement_plan.get("instances") or ()
    result = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("status") != "planned":
            raise CandidateFailure(
                "placement", "static_source_placement_rejected"
            )
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise CandidateFailure(
                "placement", "static_source_placement_instance_id_missing"
            )
        if instance_id in result:
            raise CandidateFailure(
                "placement", "static_source_placement_instance_id_repeated"
            )
        result[instance_id] = deepcopy(dict(row))
    return result


def _placement_surface_kind(placement):
    support = placement.get("support_identity") if isinstance(
        placement, Mapping
    ) else None
    return str((support or {}).get("surface_kind", "")).strip().lower()


def _placement_is_ground(placement):
    return _placement_surface_kind(placement) in {"", "floor"}


def _rotation_matrix_to_xyzw(matrix):
    value = np.asarray(matrix, dtype=float)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise CandidateFailure(
            "placement", "static_source_placement_rotation_invalid"
        )
    gram = value.T @ value
    determinant = float(np.linalg.det(value))
    if (
        not np.allclose(gram, np.eye(3), atol=1.0e-5, rtol=0.0)
        or not math.isfinite(determinant)
        or determinant <= 0.0
        or not math.isclose(determinant, 1.0, abs_tol=1.0e-5)
    ):
        raise CandidateFailure(
            "placement", "static_source_placement_rotation_not_proper"
        )
    trace = float(np.trace(value))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (value[2, 1] - value[1, 2]) / scale
        y = (value[0, 2] - value[2, 0]) / scale
        z = (value[1, 0] - value[0, 1]) / scale
    else:
        diagonal = np.diag(value)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(
                max(1.0e-12, 1.0 + value[0, 0] - value[1, 1] - value[2, 2])
            ) * 2.0
            x = 0.25 * scale
            y = (value[0, 1] + value[1, 0]) / scale
            z = (value[0, 2] + value[2, 0]) / scale
            w = (value[2, 1] - value[1, 2]) / scale
        elif index == 1:
            scale = math.sqrt(
                max(1.0e-12, 1.0 + value[1, 1] - value[0, 0] - value[2, 2])
            ) * 2.0
            x = (value[0, 1] + value[1, 0]) / scale
            y = 0.25 * scale
            z = (value[1, 2] + value[2, 1]) / scale
            w = (value[0, 2] - value[2, 0]) / scale
        else:
            scale = math.sqrt(
                max(1.0e-12, 1.0 + value[2, 2] - value[0, 0] - value[1, 1])
            ) * 2.0
            x = (value[0, 2] + value[2, 0]) / scale
            y = (value[1, 2] + value[2, 1]) / scale
            z = 0.25 * scale
            w = (value[1, 0] - value[0, 1]) / scale
    quaternion = np.asarray([x, y, z, w], dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or not math.isclose(norm, 1.0, abs_tol=1.0e-5):
        raise CandidateFailure(
            "placement", "static_source_placement_quaternion_not_unit"
        )
    quaternion = quaternion / norm
    reconstructed = _rotation_matrix_from_xyzw(quaternion)
    if not np.allclose(reconstructed, value, atol=1.0e-5, rtol=0.0):
        raise CandidateFailure(
            "placement",
            "static_source_placement_rotation_reconstruction_mismatch",
        )
    return quaternion.tolist()


def _quaternions_equivalent(left, right):
    try:
        first = np.asarray(left, dtype=float)
        second = np.asarray(right, dtype=float)
    except (TypeError, ValueError):
        return False
    return (
        first.shape == (4,)
        and second.shape == (4,)
        and np.all(np.isfinite(first))
        and np.all(np.isfinite(second))
        and (
            np.allclose(first, second, atol=1.0e-5, rtol=0.0)
            or np.allclose(first, -second, atol=1.0e-5, rtol=0.0)
        )
    )


def _rotation_matrix_from_xyzw(quaternion):
    x, y, z, w = (
        float(value) for value in np.asarray(quaternion, dtype=float)
    )
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _root_matrix_from_pose(translation, quaternion):
    rotation = _rotation_matrix_from_xyzw(quaternion)
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(translation, dtype=float)
    return matrix.reshape(-1).tolist()


def _pins_for_actors(profile, actors):
    """Align ``profile['pinned_static_positions_m']`` to this Episode's actors.

    The caller states the pins against the identity it declared - an entity
    instance id, or the source slot the room bank names them by - so the same
    configuration keeps meaning the same body when the draw order changes. A
    name that no actor carries is refused rather than dropped, because a pin
    that quietly does nothing is worse than one that fails.
    """
    declared = profile.get('pinned_static_positions_m')
    if declared is None:
        return None
    if not isinstance(declared, Mapping):
        raise ValueError(
            'pinned_static_positions_m must map an entity instance or source slot '
            'to a three metre position')
    index_by_name = {}
    for index, actor in enumerate(actors):
        for name in (actor.get('entity_instance_id'), actor.get('source_slot_id'),
                     actor.get('actor_id')):
            if name:
                index_by_name.setdefault(str(name), index)
    unknown = sorted(set(map(str, declared)) - set(index_by_name))
    if unknown:
        raise ValueError(
            'pinned_static_positions_m names ' + ', '.join(unknown)
            + '; this Episode declares ' + ', '.join(sorted(index_by_name)))
    pins = [None] * len(actors)
    for name, value in declared.items():
        pins[index_by_name[str(name)]] = value
    return pins


# Which C05 control words carry a real measured support. Anything else is a
# missing measurement, and a missing measurement is refused rather than filled
# in with a constant.
MEASURED_CONTACT_CONTROLS = ('native_measured_sole', 'cpu_measured_sole')
MEASURED_CONTACT_EVIDENCE = ('native_executed_pose', 'cpu_reconstruction', 'cpu_reconstruction_from_baked_clips')


def _contact_document(declared):
    """Read a C05 contact document, retaining trajectory-query metadata."""
    if declared is None:
        return None
    if isinstance(declared, str):
        declared = read_json_document(declared)
    if not isinstance(declared, Mapping):
        raise ValueError('contact_correction must be a mapping or a path to one')
    if isinstance(declared.get('document'), Mapping):
        document = dict(declared['document'])
        document.update({
            key: value for key, value in declared.items()
            if key != 'document'
        })
        return document
    if declared.get('path'):
        document = read_json_document(str(declared['path']))
        if not isinstance(document, Mapping):
            raise ValueError('contact_correction path must contain an object')
        document = dict(document)
        document.update({
            key: value for key, value in declared.items()
            if key != 'path'
        })
        return document
    return dict(declared)


def _contact_controls(declared):
    """Read C05's contact_correction document, inline or by path."""
    document = _contact_document(declared)
    if document is None:
        return None
    source = document.get('controls')
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        raise ValueError('contact_correction carries no controls list')
    return list(source)


def read_json_document(path):
    import json as _json
    with open(path, encoding='utf-8') as handle:
        return _json.load(handle)


def _contact_identity(document, control):
    """Return the identity declared by the request and reject mixed-world rows."""
    identity = document.get('identity') or document.get('world_identity') or {}
    local = control.get('identity') or {}
    if not isinstance(identity, Mapping) or not isinstance(local, Mapping):
        raise ValueError('trajectory contact correction identity must be a mapping')
    merged = dict(identity)
    for key, value in local.items():
        if key in merged and merged[key] is not None and value is not None:
            if str(merged[key]) != str(value):
                raise CandidateFailure(
                    'routes',
                    f'contact_correction_identity_mismatch:{key}',
                )
        elif key not in merged or merged[key] is None:
            merged[key] = value
    required = ('episode_id', 'room_id')
    if any(not merged.get(key) for key in required):
        raise CandidateFailure(
            'routes',
            'contact_correction_identity_missing_episode_or_room',
        )
    # A null world_id is meaningful: a stage directory alias must not be
    # promoted into a synthetic identity.
    merged.setdefault('world_id', None)
    return merged


def _contact_geometry_registration(document, identity):
    """Load the room mesh and navigation layer registered by this request."""
    geometry = document.get('room_visual_geometry') or document.get('geometry')
    navigation = document.get('navigation_floor') or document.get('navigation')
    if not isinstance(geometry, Mapping):
        raise CandidateFailure(
            'routes', 'contact_correction_missing_room_visual_geometry'
        )
    if not isinstance(navigation, Mapping):
        raise CandidateFailure(
            'routes', 'contact_correction_missing_navigation_floor_registration'
        )
    scene_glb = geometry.get('scene_glb') or geometry.get('path')
    dataset_config = (
        geometry.get('scene_dataset_config')
        or geometry.get('dataset_config')
    )
    if not scene_glb or not dataset_config:
        raise CandidateFailure(
            'routes',
            'contact_correction_room_visual_geometry_requires_scene_glb_and_dataset_config',
        )
    floor = navigation.get('floor_height_m', navigation.get('navigation_floor_m'))
    floor_source = navigation.get(
        'source', navigation.get('navigation_floor_source')
    )
    if floor is None or not floor_source:
        raise CandidateFailure(
            'routes',
            'contact_correction_navigation_floor_requires_measured_height_and_source',
        )
    try:
        floor = float(floor)
    except (TypeError, ValueError) as exc:
        raise CandidateFailure(
            'routes', 'contact_correction_navigation_floor_is_not_finite'
        ) from exc
    if not math.isfinite(floor):
        raise CandidateFailure(
            'routes', 'contact_correction_navigation_floor_is_not_finite'
        )
    registered_identity = geometry.get('identity') or navigation.get('identity')
    if registered_identity is not None:
        if not isinstance(registered_identity, Mapping):
            raise ValueError('registered contact geometry identity must be a mapping')
        for key in ('episode_id', 'room_id', 'scene_id', 'world_id'):
            expected = identity.get(key)
            observed = registered_identity.get(key)
            if expected is not None and observed is not None and str(expected) != str(observed):
                raise CandidateFailure(
                    'routes', f'contact_correction_geometry_identity_mismatch:{key}'
                )
    # This import is deliberately on the configured path only. Ordinary plans
    # without trajectory contact correction never load the qualification mesh.
    from avengine.assets import qualification_geometry as _geometry
    try:
        vertices, triangles, evidence = _geometry.load_room_triangles(
            scene_glb,
            scene_dataset_config=dataset_config,
        )
    except Exception as exc:
        raise CandidateFailure(
            'routes',
            'contact_correction_room_visual_geometry_load_failed:' + str(exc),
        ) from exc
    return {
        'room_vertices': np.asarray(vertices, dtype=float),
        'room_triangles': np.asarray(triangles, dtype=np.int64),
        'scene_glb': str(scene_glb),
        'scene_dataset_config': str(dataset_config),
        'scene_ref': str(geometry.get('scene_ref') or scene_glb),
        'geometry_version': geometry.get('geometry_version'),
        'navigation_floor_m': floor,
        'navigation_floor_source': str(floor_source),
        'navmesh': navigation.get('navmesh'),
        'identity': deepcopy(dict(identity)),
        'loader_evidence': {
            key: deepcopy(evidence.get(key))
            for key in (
                'frame', 'stage_axes', 'vertex_count', 'triangle_count',
                'source_byte_size', 'source_sha256',
            )
            if key in evidence
        },
    }


def _contact_foot_row(control):
    """Read one measured actor row from C05's native foot-contact evidence."""
    row = control.get('foot_contact')
    path = control.get('foot_contact_path')
    if path:
        try:
            payload = read_json_document(str(path))
        except (OSError, ValueError) as exc:
            raise CandidateFailure(
                'routes', 'contact_correction_foot_contact_read_failed:' + str(exc)
            ) from exc
        rows = payload.get('actors') if isinstance(payload, Mapping) else None
        wanted = str(control.get('foot_contact_actor_id') or control.get('actor_id'))
        row = next(
            (
                candidate for candidate in rows or ()
                if isinstance(candidate, Mapping)
                and str(candidate.get('actor_id') or candidate.get('entity_instance_id')) == wanted
            ),
            None,
        )
        if row is None:
            raise CandidateFailure(
                'routes', 'contact_correction_foot_contact_actor_missing:' + wanted
            )
    if row is None:
        return None
    if not isinstance(row, Mapping):
        raise ValueError('foot_contact must be a mapping or a path to one')
    return row


def _contact_offset_and_footprint(control):
    """Resolve measured sole evidence and its measured footprint."""
    from avengine.assets import qualification_geometry as _geometry

    foot_row = _contact_foot_row(control)
    derived = None
    if foot_row is not None:
        derived = _geometry.articulated_contact_offset(foot_row)
    supplied = control.get('contact_offset')
    offset = dict(supplied) if isinstance(supplied, Mapping) else derived
    if offset is None:
        raise CandidateFailure(
            'routes', 'contact_correction_missing_measured_foot_contact'
        )
    if derived is not None and derived.get('measurement') == 'measured':
        if supplied is not None:
            try:
                mismatch = abs(
                    float(supplied.get('root_above_contact_m'))
                    - float(derived.get('root_above_contact_m'))
                ) > 1.0e-6
            except (TypeError, ValueError):
                mismatch = True
            if mismatch:
                raise CandidateFailure(
                    'routes', 'contact_correction_foot_evidence_offset_mismatch'
                )
        offset = dict(derived)
    footprint = control.get('footprint_extent_m')
    if footprint is None and foot_row is not None:
        frames = foot_row.get('frames') or ()
        if frames:
            footprint = frames[0].get('sole_footprint_extent_m')
    if not isinstance(footprint, Sequence) or isinstance(footprint, (str, bytes)):
        raise CandidateFailure(
            'routes', 'contact_correction_missing_measured_footprint'
        )
    try:
        footprint = [float(value) for value in footprint[:2]]
    except (TypeError, ValueError) as exc:
        raise CandidateFailure(
            'routes', 'contact_correction_invalid_measured_footprint'
        ) from exc
    if len(footprint) != 2 or any(not math.isfinite(value) or value <= 0 for value in footprint):
        raise CandidateFailure(
            'routes', 'contact_correction_invalid_measured_footprint'
        )
    return offset, footprint, foot_row


def _contact_yaw_series(actor, path, control):
    """Return the footprint yaw used for support queries at each route point."""
    frames = len(path)
    declared = control.get('yaw_deg')
    if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes)):
        if len(declared) != frames:
            raise CandidateFailure(
                'routes', 'contact_correction_yaw_series_frame_count_mismatch'
            )
        values = np.asarray(declared, dtype=float)
        if not np.all(np.isfinite(values)):
            raise CandidateFailure('routes', 'contact_correction_yaw_series_not_finite')
        return values, 'declared_per_frame'
    if declared is not None:
        try:
            value = float(declared)
        except (TypeError, ValueError) as exc:
            raise CandidateFailure('routes', 'contact_correction_yaw_not_finite') from exc
        if not math.isfinite(value):
            raise CandidateFailure('routes', 'contact_correction_yaw_not_finite')
        return np.full(frames, value, dtype=float), 'declared'
    timeline = actor.get('timeline') or {}
    anatomical = np.asarray(
        timeline.get('local_anatomical_forward_axis', [1.0, 0.0, 0.0]),
        dtype=float,
    )
    anatomical_yaw = math.atan2(-float(anatomical[2]), float(anatomical[0]))
    result = np.zeros(frames, dtype=float)
    if frames > 1:
        delta = np.diff(np.asarray(path, dtype=float), axis=0)
        for index, vector in enumerate(delta, start=1):
            if np.linalg.norm(vector[[0, 2]]) > 1.0e-12:
                result[index] = math.degrees(
                    math.atan2(-float(vector[2]), float(vector[0])) - anatomical_yaw
                )
            else:
                result[index] = result[index - 1]
    return result, 'trajectory_direction'


def _apply_trajectory_contact_correction(actors, paths, document, room=None, planned_yaw_series=None):
    """Query C05 support at every configured route point before derivation."""
    identity = _contact_identity(document, {})
    if isinstance(room, Mapping) and room.get('room_id'):
        if str(room['room_id']) != str(identity['room_id']):
            raise CandidateFailure(
                'routes', 'contact_correction_identity_room_mismatch'
            )
    registration = _contact_geometry_registration(document, identity)
    controls = list(document.get('controls') or ())
    by_name = {}
    for control in controls:
        if not isinstance(control, Mapping):
            raise ValueError('each contact_correction control must be a mapping')
        for key in ('actor_id', 'entity_instance_id', 'asset_id'):
            name = control.get(key)
            if name:
                by_name.setdefault(str(name), []).append(control)
    query_defaults = document.get('support_query') or {}
    if not isinstance(query_defaults, Mapping):
        raise ValueError('contact_correction support_query must be a mapping')
    from avengine.assets import qualification_geometry as _geometry

    query_cache = {}
    query_count = 0
    cache_hits = 0
    pending = []
    applied = []
    for index, actor in enumerate(actors):
        names = [
            str(actor.get(key)) for key in (
                'entity_instance_id', 'actor_id', 'source_slot_id', 'asset_id'
            ) if actor.get(key)
        ]
        matches = [control for name in names for control in by_name.get(name, ())]
        if not matches:
            pending.append(None)
            continue
        control = matches[0]
        control_identity = _contact_identity(document, control)
        if control_identity != identity:
            raise CandidateFailure(
                'routes', 'contact_correction_control_identity_mismatch'
            )
        if str(control.get('control')) not in MEASURED_CONTACT_CONTROLS:
            raise CandidateFailure(
                'routes',
                'contact_correction_without_measured_support:'
                + str(actor.get('entity_instance_id')),
            )
        offset, footprint, foot_row = _contact_offset_and_footprint(control)
        if (
            offset.get('measurement') != 'measured'
            or str(offset.get('evidence_kind')) not in MEASURED_CONTACT_EVIDENCE
            or offset.get('root_above_contact_m') is None
        ):
            raise CandidateFailure(
                'routes',
                'contact_correction_without_measured_support:'
                + str(actor.get('entity_instance_id')),
            )
        path = np.asarray(paths[index], dtype=float)
        if path.ndim != 2 or path.shape[1] != 3:
            raise CandidateFailure(
                'routes', 'contact_correction_invalid_trajectory_shape:' + str(index)
            )
        if planned_yaw_series is not None and planned_yaw_series[index] is not None:
            yaws = np.asarray(planned_yaw_series[index], dtype=float)
            yaw_source = 'same_as_planned_root_rotation'
        else:
            yaws, yaw_source = _contact_yaw_series(actor, path, control)
        options = dict(query_defaults)
        options.update(control.get('support_query') or {})
        support_statistic = str(
            options.get('support_statistic', 'support_top_q95_m')
        )
        target_gap = float(options.get('target_gap_m', 0.0))
        search_below_floor = float(options.get('search_below_floor_m', 0.0))
        if not math.isfinite(search_below_floor) or search_below_floor < 0.0:
            raise CandidateFailure(
                'routes', 'contact_correction_invalid_search_below_floor_m'
            )
        actor_pending = []
        support_rows = []
        unique_actor_queries = 0
        actor_cache_hits = 0
        for frame_index, (point, yaw) in enumerate(zip(path, yaws)):
            search_ceiling = options.get('search_ceiling_m')
            if search_ceiling is not None:
                search_ceiling = float(search_ceiling)
            cache_key = (
                str(registration['scene_glb']),
                str(registration['scene_dataset_config']),
                tuple(round(float(value), 7) for value in point),
                tuple(round(float(value), 7) for value in footprint),
                round(float(yaw), 7),
                round(float(registration['navigation_floor_m']), 7),
                str(registration['navigation_floor_source']),
                search_ceiling,
                search_below_floor,
                target_gap,
                support_statistic,
                tuple(sorted((str(k), repr(v)) for k, v in offset.items())),
                int(options.get('grid', 9)),
                float(options.get('normal_tilt_limit_deg', 20.0)),
                float(options.get('level_cluster_m', 0.01)),
                float(options.get('coverage_fraction', 0.9)),
            )
            cached = query_cache.get(cache_key)
            cache_hit = cached is not None
            if cache_hit:
                cache_hits += 1
                actor_cache_hits += 1
                query, correction = cached
            else:
                query_count += 1
                unique_actor_queries += 1
                query = _geometry.query_support_under_footprint(
                    room_vertices=registration['room_vertices'],
                    room_triangles=registration['room_triangles'],
                    centre_world_m=point,
                    footprint_extent_m=footprint,
                    navigation_floor_m=registration['navigation_floor_m'],
                    navigation_floor_source=registration['navigation_floor_source'],
                    yaw_deg=float(yaw),
                    search_ceiling_m=search_ceiling,
                    search_below_floor_m=search_below_floor,
                    grid=int(options.get('grid', 9)),
                    normal_tilt_limit_deg=float(
                        options.get('normal_tilt_limit_deg', 20.0)
                    ),
                    level_cluster_m=float(options.get('level_cluster_m', 0.01)),
                    coverage_fraction=float(options.get('coverage_fraction', 0.9)),
                    scene_ref=registration['scene_ref'],
                )
                correction = _geometry.plan_root_contact_correction(
                    support_query=query,
                    contact_offset=offset,
                    current_root_world_m=point,
                    target_gap_m=target_gap,
                    identity=identity,
                    support_statistic=support_statistic,
                )
                query_cache[cache_key] = (query, correction)
            if correction.get('measurement') == 'not_run':
                raise CandidateFailure(
                    'routes',
                    'contact_correction_without_measured_support:'
                    + str(actor.get('entity_instance_id'))
                    + f':frame_{frame_index}',
                )
            corrected = correction.get('corrected_root_world_m')
            if not isinstance(corrected, Sequence) or len(corrected) != 3:
                raise CandidateFailure(
                    'routes',
                    'contact_correction_without_corrected_root:'
                    + str(actor.get('entity_instance_id'))
                    + f':frame_{frame_index}',
                )
            corrected_y = float(corrected[1])
            if not math.isfinite(corrected_y):
                raise CandidateFailure(
                    'routes', 'contact_correction_corrected_height_not_finite'
                )
            supporting = query.get('supporting_level') or {}
            support_rows.append({
                'frame_index': int(frame_index),
                'cache_hit': bool(cache_hit),
                'centre_world_m': [float(value) for value in point],
                'yaw_deg': float(yaw),
                'support_level_height_m': float(
                    correction['support_level_height_m']
                ),
                'support_level_coverage_fraction': supporting.get(
                    'footprint_coverage_fraction'
                ),
                'support_level_normal_m': deepcopy(
                    correction.get('support_level_normal_m')
                ),
                'corrected_root_height_m': corrected_y,
                'root_delta_m': float(correction['root_delta_m']),
                'prediction': correction.get('prediction'),
                'native_verification': correction.get('native_verification'),
            })
            actor_pending.append(corrected_y)
        pending.append(np.asarray(actor_pending, dtype=float))
        applied.append({
            'entity_instance_id': actor.get('entity_instance_id'),
            'actor_id': actor.get('actor_id'),
            'control': str(control.get('control')),
            'evidence_kind': str(offset.get('evidence_kind')),
            'contact_offset': deepcopy(offset),
            'foot_contact_evidence': {
                'path': control.get('foot_contact_path'),
                'actor_id': control.get('foot_contact_actor_id') or control.get('actor_id'),
                'measurement': foot_row.get('measurement') if foot_row else None,
                'native_frame_count': foot_row.get('native_frame_count') if foot_row else None,
            },
            'footprint_extent_m': list(footprint),
            'yaw_source': yaw_source,
            'frame_count': len(path),
            'support_level_height_series_m': [
                row['support_level_height_m'] for row in support_rows
            ],
            'corrected_root_height_series_m': list(map(float, actor_pending)),
            'root_delta_series_m': [row['root_delta_m'] for row in support_rows],
            'support_query_evidence': support_rows,
            'unique_query_count': unique_actor_queries,
            'cache_hit_count': actor_cache_hits,
            'identity': deepcopy(identity),
        })
    for index, corrected in enumerate(pending):
        if corrected is not None:
            paths[index][:, 1] = corrected
    return {
        'applied': applied,
        'per_trajectory_point': True,
        'query_count': int(query_count),
        'cache_hit_count': int(cache_hits),
        'geometry': {
            key: deepcopy(registration[key])
            for key in (
                'scene_glb', 'scene_dataset_config', 'scene_ref',
                'geometry_version', 'navigation_floor_m',
                'navigation_floor_source', 'navmesh', 'identity',
                'loader_evidence',
            )
        },
        'stage': 'canonical_plan_before_rotation_emitter_and_body_derivation',
        'authority': 'avengine_c05_root_contact_correction_v1_per_trajectory_point',
        'claim_boundary': (
            'each corrected height is a CPU prediction from the registered room mesh, '
            'navigation floor and measured foot evidence; no native capture has '
            'confirmed the resulting trajectory contact'
        ),
    }


def apply_contact_correction(actors, paths, profile, *, room=None, planned_yaw_series=None):
    """Put each corrected body on its measured visual support, before anything derives from it.

    C05 measures two things: the level the visual mesh actually supports the
    body at, and how far that asset's root sits above its own contact point.
    The corrected root height is their sum. This runs on the sampled routes,
    before rotations, emitters and body proxies are derived from them, because
    C05's own contract says a height applied to a finished plan leaves the rest
    of that plan stale. Moving the root here means the emitter, the body proxy,
    the acoustic inputs and every recorded frame all come from the corrected
    position.

    A control without a measured support is refused. Filling one in with a
    constant would put a body on a floor nobody measured.
    """
    document = _contact_document(profile.get('contact_correction'))
    if not document:
        return None
    if document.get('query_support_under_footprint') in (
        'per_trajectory_sample', 'per_trajectory_point', True
    ) or str(document.get('schema', '')).startswith(
        'avengine_c01r4_trajectory_contact_correction_'
    ):
        return _apply_trajectory_contact_correction(
            actors, paths, document, room=room, planned_yaw_series=planned_yaw_series
        )
    controls = document.get('controls')
    if not isinstance(controls, Sequence) or isinstance(controls, (str, bytes)):
        raise ValueError('contact_correction carries no controls list')
    controls = list(controls)
    by_name = {}
    for control in controls:
        for key in ('actor_id', 'entity_instance_id', 'asset_id'):
            name = control.get(key)
            if name:
                by_name.setdefault(str(name), []).append(control)
    applied, refused = [], []
    for index, actor in enumerate(actors):
        names = [str(actor.get(key)) for key in
                 ('entity_instance_id', 'actor_id', 'source_slot_id', 'asset_id')
                 if actor.get(key)]
        control = next((by_name[name][0] for name in names if name in by_name), None)
        if control is None:
            continue
        offset = control.get('contact_offset') or {}
        correction = control.get('correction') or {}
        support = correction.get('support_level_height_m')
        root_above = offset.get('root_above_contact_m')
        if (str(control.get('control')) not in MEASURED_CONTACT_CONTROLS
                or str(offset.get('evidence_kind')) not in MEASURED_CONTACT_EVIDENCE
                or support is None or root_above is None):
            refused.append({
                'entity_instance_id': actor.get('entity_instance_id'),
                'control': control.get('control'),
                'evidence_kind': offset.get('evidence_kind'),
                'reason': 'no measured visual support for this body',
            })
            continue
        target_y = float(support) + float(root_above)
        before = float(paths[index][0][1])
        paths[index][:, 1] = target_y
        applied.append({
            'entity_instance_id': actor.get('entity_instance_id'),
            'control': str(control.get('control')),
            'evidence_kind': str(offset.get('evidence_kind')),
            'support_level_height_m': float(support),
            'root_above_contact_m': float(root_above),
            'corrected_root_height_m': target_y,
            'root_height_before_m': before,
            'root_delta_m': target_y - before,
        })
    if refused:
        raise CandidateFailure(
            'routes',
            'contact_correction_without_measured_support:'
            + ','.join(str(row['entity_instance_id']) for row in refused))
    return {
        'applied': applied,
        'stage': 'canonical_plan_before_rotation_emitter_and_body_derivation',
        'authority': 'avengine_c05_root_contact_correction_v1',
        'claim_boundary': ('the corrected height is arithmetic on C05 measured inputs; no '
                           'native capture has confirmed the resulting contact'),
    }


def sample_routes(space, actors, profile, clock, rng, region=None, *,
                 required_windows=None, room=None, motion_requirements=None,
                 motion_budget=None, static_placements=None,
                 constructive_motion=False,
                 visibility_requirements=()):
    frames, fps = int(clock['frame_count']), float(clock['frame_rate_hz'])
    flags = _moving_flags(profile, actors, rng, motion_requirements)
    visibility_only_movers: set[int] = set()
    # A pixel visibility requirement that needs a state *change* cannot be met
    # by a still route under a fixed camera. conditioned_visibility knows which
    # ones those are and used to say so only after the routes were drawn, when
    # the only outcome left was a refusal. Ask it before drawing instead.
    if visibility_requirements and motion_requirements is None:
        from avengine.rooms import conditioned_visibility as _cv
        mobile = [
            str(actor['entity_instance_id'])
            for actor in actors
            if actor['entity_class'] not in RIGID
        ]
        motion_plan = _cv.route_motion_plan(
            visibility_requirements, mobile_instance_ids=mobile)
        if motion_plan['unsatisfiable']:
            raise CandidateFailure(
                'routes', 'visibility_requirement_needs_motion_no_mobile_instance')
        index_by_id = {
            str(actor['entity_instance_id']): index
            for index, actor in enumerate(actors)
        }
        for instance_id in motion_plan['must_move']:
            index = index_by_id.get(instance_id)
            if index is None:
                raise CandidateFailure(
                    'routes', 'visibility_requirement_names_unknown_instance')
            if actors[index]['entity_class'] in RIGID:
                raise CandidateFailure(
                    'routes', 'visibility_requirement_needs_motion_from_static_entity')
            if not flags[index]:
                # No motion knob asked for this one, so its walk carries no
                # claim about the audible window.
                visibility_only_movers.add(index)
            flags[index] = True
        # Keep the in-memory profile JSON-native. A tuple serializes as a list
        # in the snapshot, which otherwise makes the returned plan and its
        # persisted condition_profile compare unequal.
        profile['visibility_forced_movers'] = list(motion_plan['must_move'])
        profile['visibility_route_motion_plan'] = {
            key: (
                list(motion_plan[key])
                if key == 'must_move' else motion_plan[key]
            )
            for key in ('must_move', 'rows', 'unsatisfiable')
        }
    required_windows=required_windows or {}
    required_frames=max([1]+[int(math.ceil((b-a)*fps/clock['sample_rate_hz']))+1 for a,b in required_windows.values()])
    placement_rows = _static_placement_rows(static_placements)
    constructive_motion_active = bool(
        constructive_motion and motion_requirements
    )
    if constructive_motion_active and not any(
        not _placement_is_ground(row) for row in placement_rows.values()
    ):
        if space.route_bank() is not None:
            requirements_by_index = {
                index: motion_requirements.get(
                    actor.get('entity_instance_id')
                )
                for index, actor in enumerate(actors)
            }
            paths, metadata = _native_routes(
                space, flags, frames, fps, rng,
                start_hold_frames=profile.get('native_start_hold_frames'),
                pinned_static_positions_m=_pins_for_actors(profile, actors),
                constructive_motion_requirements=requirements_by_index,
            )
            metadata['motion_construction'] = (
                'constructive_existing_native_route_bank'
            )
            ok, floor_y = _points_same_floor(
                [p[0] for p in paths] + [p[-1] for p in paths]
            )
            if not ok:
                raise CandidateFailure(
                    'routes', 'native_routes_not_on_same_floor'
                )
            floor_source = 'constructive_existing_native_route_bank'
            metadata = {
                **metadata,
                'selected_floor_height_m': floor_y,
                'same_floor_tolerance_m': SAME_FLOOR_Y_TOLERANCE_M,
            }
        else:
            (
                paths,
                metadata,
                floor_y,
                floor_source,
            ) = _constructive_motion_paths(
                space, actors, profile, clock, rng, region, room,
                motion_requirements, motion_budget,
            )
    elif space.route_bank() is not None:
        if any(
            not _placement_is_ground(row)
            for row in placement_rows.values()
        ):
            raise CandidateFailure(
                'routes', 'non_ground_static_source_requires_non_native_route_adapter'
            )
        paths, metadata = _native_routes(
            space, flags, frames, fps, rng,
            start_hold_frames=profile.get('native_start_hold_frames'),
            pinned_static_positions_m=_pins_for_actors(profile, actors))
        ok, floor_y = _points_same_floor([p[0] for p in paths] + [p[-1] for p in paths])
        if not ok:
            raise CandidateFailure('routes', 'native_routes_not_on_same_floor')
        floor_source = 'native_route_paths'
        metadata = {**metadata, 'selected_floor_height_m': floor_y,
                    'same_floor_tolerance_m': SAME_FLOOR_Y_TOLERANCE_M}
        metadata.setdefault('motion_construction', 'legacy_random_route_then_solver')
    else:
        if placement_rows and all(
            not _placement_is_ground(row) for row in placement_rows.values()
        ):
            floor_y, floor_source = static_support_floor_reference(
                room, space, placement_rows
            )
            floor_region = space.bounds().copy() if region is None else np.asarray(
                region, dtype=float
            ).copy()
            floor_region[0, 1] = floor_y - SAME_FLOOR_Y_TOLERANCE_M
            floor_region[1, 1] = floor_y + SAME_FLOOR_Y_TOLERANCE_M
        else:
            floor_region, floor_y = lock_same_floor_region(space, rng, region, room)
            floor_source = 'declared_or_sampled_navigation_floor'
        paths=[]; records=[]; hub=space.sample_navigable(rng, floor_region)
        if abs(float(hub[1]) - floor_y) > SAME_FLOOR_Y_TOLERANCE_M:
            raise CandidateFailure('routes', 'hub_left_selected_floor')
        bounds=floor_region.copy()
        bounds[0,[0,2]]=np.maximum(bounds[0,[0,2]],hub[[0,2]]-3.1)
        bounds[1,[0,2]]=np.minimum(bounds[1,[0,2]],hub[[0,2]]+3.1)
        bounds[0,1]=floor_region[0,1]; bounds[1,1]=floor_region[1,1]
        for i, required_motion in enumerate(flags):
            actor_required_frames = (
                1 if i in visibility_only_movers else required_frames
            )
            placement = placement_rows.get(
                actors[i].get('entity_instance_id')
            )
            if placement is not None and not _placement_is_ground(placement):
                if required_motion:
                    raise CandidateFailure(
                        'routes', 'non_ground_static_source_cannot_move'
                    )
                root = np.asarray(
                    placement['root_transform']['translation_m'],
                    dtype=float,
                )
                if root.shape != (3,) or not np.all(np.isfinite(root)):
                    raise CandidateFailure(
                        'placement', 'static_source_placement_translation_invalid'
                    )
                # Support-catalog placement owns non-ground peer fit. The
                # navigation separation gate applies only to ground actors.
                route=np.repeat(root[None], frames, axis=0)
                record={
                    'motion':'static',
                    'route_points_m':None,
                    'placement_status':'planned',
                    'support_identity':deepcopy(
                        placement.get('support_identity') or {}
                    ),
                    'navigation_authority':'support_surface_not_ground',
                }
                paths.append(route)
                records.append(record)
                continue
            start=space.sample_navigable(rng, bounds)
            if abs(float(start[1]) - floor_y) > SAME_FLOOR_Y_TOLERANCE_M:
                raise CandidateFailure('routes', 'placement_left_selected_floor')
            if any(
                _placement_is_ground(
                    placement_rows.get(
                        actors[index].get('entity_instance_id')
                    )
                )
                and np.linalg.norm(start-p[0]) < .95
                for index,p in enumerate(paths)
            ):
                raise CandidateFailure('routes','initial_source_separation_below_0.95_m')
            route=np.repeat(start[None], frames, axis=0); record={'motion':'static','route_points_m':None}
            if required_motion:
                end=space.sample_navigable(rng,bounds)
                if abs(float(end[1]) - floor_y) > SAME_FLOOR_Y_TOLERANCE_M:
                    raise CandidateFailure('routes', 'placement_left_selected_floor')
                poly=space.shortest_path(start,end)
                if poly is None or len(poly)<2:
                    raise CandidateFailure('routes','no_existing_navigation_path')
                length=float(np.linalg.norm(np.diff(poly,axis=0),axis=1).sum())
                if length < max(1.5,(actor_required_frames-1)/fps*.5):
                    raise CandidateFailure('routes','path_too_short_for_moving_window')
                low_speed,high_speed=profile.get('walk_speed_range_mps',(.5,.8))
                speed=float(rng.uniform(float(low_speed),float(high_speed))) if high_speed>low_speed else float(low_speed)
                moving_frames=max(2,int(math.ceil(length/speed*fps))+1)
                if moving_frames>=frames or moving_frames<actor_required_frames:
                    raise CandidateFailure('routes','route_does_not_fit_clock_or_required_window')
                pause=int(rng.integers(max(1,int(fps)),max(2,int(2*fps))+1))
                modes=['walk_with_sampled_holds']
                if moving_frames>=2*actor_required_frames and moving_frames+pause<frames:modes.append('walk_with_sampled_pause')
                mode=modes[int(rng.integers(len(modes)))];extra=pause if mode=='walk_with_sampled_pause' else 0
                start_frame=int(rng.integers(frames-moving_frames-extra+1));motion=resample_polyline_by_arc_length(poly,moving_frames)
                if extra:
                    split=int(rng.integers(actor_required_frames,moving_frames-actor_required_frames+1))
                    motion=np.concatenate([motion[:split],np.repeat(motion[split-1:split],pause,axis=0),motion[split:]])
                route[start_frame:start_frame+len(motion)]=motion;route[start_frame+len(motion):]=motion[-1]
                record={'motion':mode,'start_frame':start_frame,'end_frame_exclusive':start_frame+len(motion),'route_points_m':poly.tolist(),
                        'required_contiguous_motion_frames':actor_required_frames}
            if not all(space.is_navigable(p) for p in route):
                raise CandidateFailure('routes','sampled_path_left_existing_navigation')
            if any(
                _placement_is_ground(
                    placement_rows.get(
                        actors[index].get('entity_instance_id')
                    )
                )
                and np.linalg.norm(route-p,axis=1).min()<.95
                for index,p in enumerate(paths)
            ):
                raise CandidateFailure('routes','all_frame_source_separation_below_0.95_m')
            ok, _ = _points_same_floor(route, floor_y)
            if not ok:
                raise CandidateFailure('routes', 'sampled_path_left_selected_floor')
            paths.append(route);records.append(record)
        metadata={'authority':space.metadata['authority'],'actors':records,'minimum_separation_m':.95,
                  'selected_floor_height_m': floor_y, 'same_floor_tolerance_m': SAME_FLOOR_Y_TOLERANCE_M}
        metadata['motion_construction'] = 'legacy_random_route_then_solver'
    if motion_requirements:
        if motion_budget is None:
            raise CandidateFailure('routes', 'motion_solver_budget_unavailable')
        _apply_motion_requirements(
            space, actors, paths, metadata, clock, motion_requirements, motion_budget)
    moving_threshold = (
        float(motion_budget.moving_threshold_mps)
        if motion_budget is not None else .05
    )
    # Draw headings once, in the same actor order as root materialization.
    # Contact must query the footprint in the orientation the renderer will use.
    initial_headings = []
    planned_yaws = []
    for actor, path in zip(actors, paths):
        placement = placement_rows.get(actor.get('entity_instance_id'))
        if placement is not None and not _placement_is_ground(placement):
            initial_headings.append(None)
            planned_yaws.append(None)
            continue
        heading = float(rng.uniform(-math.pi, math.pi))
        initial_headings.append(heading)
        delta = np.diff(path, axis=0)
        delta = np.concatenate([delta, delta[-1:]], axis=0)
        mask = np.linalg.norm(delta, axis=1) * fps > moving_threshold
        anatomical = np.asarray((actor.get('timeline') or {}).get(
            'local_anatomical_forward_axis', [1., 0., 0.]))
        anatomical_yaw = math.atan2(-float(anatomical[2]), float(anatomical[0]))
        yaws = []
        for vector, is_moving in zip(delta, mask):
            if is_moving:
                heading = math.atan2(-float(vector[2]), float(vector[0])) - anatomical_yaw
            yaws.append(math.degrees(heading))
        planned_yaws.append(yaws)
    contact_report = apply_contact_correction(
        actors, paths, profile, room=room, planned_yaw_series=planned_yaws)
    if contact_report is not None:
        metadata = {**metadata, 'contact_correction': contact_report}
    moving=[]; rotations=[]; emitters=[]; bodies=[]
    for actor_index, (actor,path) in enumerate(zip(actors,paths)):
        placement = placement_rows.get(
            actor.get('entity_instance_id')
        )
        if placement is not None and not _placement_is_ground(placement):
            matrix = np.asarray(
                placement['root_transform'].get('matrix_row_major'),
                dtype=float,
            )
            if matrix.shape != (16,) or not np.all(np.isfinite(matrix)):
                raise CandidateFailure(
                    'placement', 'static_source_placement_matrix_invalid'
                )
            matrix = matrix.reshape(4, 4)
            translation = matrix[:3, 3]
            if not np.allclose(
                path,
                np.repeat(translation[None], frames, axis=0),
                atol=1.0e-8,
                rtol=0.0,
            ):
                raise CandidateFailure(
                    'placement', 'static_source_placement_path_mismatch'
                )
            quaternion = _rotation_matrix_to_xyzw(matrix[:3, :3])
            declared_quaternion = (
                placement['root_transform'].get('rotation_xyzw')
            )
            if (
                declared_quaternion is not None
                and not _quaternions_equivalent(
                    declared_quaternion, quaternion
                )
            ):
                raise CandidateFailure(
                    'placement',
                    'static_source_placement_root_quaternion_mismatch',
                )
            declared_emitter_quaternion = (
                placement['emitter_transform'].get('rotation_xyzw')
            )
            if (
                declared_emitter_quaternion is not None
                and not _quaternions_equivalent(
                    declared_emitter_quaternion, quaternion
                )
            ):
                raise CandidateFailure(
                    'placement',
                    'static_source_placement_emitter_quaternion_mismatch',
                )
            rotation = matrix[:3, :3]
            emitter_position = np.asarray(
                placement['emitter_transform']['position_m'],
                dtype=float,
            )
            if emitter_position.shape != (3,) or not np.all(
                np.isfinite(emitter_position)
            ):
                raise CandidateFailure(
                    'placement', 'static_source_placement_emitter_invalid'
                )
            offset = np.asarray(
                actor['emitter_binding']['emitter_offset_m'],
                dtype=float,
            )
            body_height = max(
                .05,
                float(
                    (placement.get('asset_resting_pose') or {}).get(
                        'height_m', offset[1] * .8
                    )
                ) * .8,
            )
            body_position = translation + rotation @ np.array(
                [0.0, body_height, 0.0]
            )
            moving.append(np.zeros(frames, dtype=bool))
            rotations.append(
                np.repeat(np.asarray(quaternion)[None], frames, axis=0)
            )
            emitters.append(
                np.repeat(emitter_position[None], frames, axis=0)
            )
            bodies.append(
                np.repeat(body_position[None], frames, axis=0)
            )
            continue
        delta=np.diff(path,axis=0); delta=np.concatenate([delta,delta[-1:]],axis=0)
        motion=np.linalg.norm(delta,axis=1)*fps > moving_threshold; moving.append(motion)
        heading=initial_headings[actor_index]; qs=[]; ep=[]; bp=[]
        offset=np.asarray(actor['emitter_binding']['emitter_offset_m'],dtype=float)
        anatomical=np.asarray((actor.get('timeline') or {}).get('local_anatomical_forward_axis',[1.,0.,0.]))
        anatomical_yaw=math.atan2(-float(anatomical[2]),float(anatomical[0]))
        for p,d,m in zip(path,delta,motion):
            if m: heading=math.atan2(-float(d[2]),float(d[0]))-anatomical_yaw
            c,s=math.cos(heading),math.sin(heading);rot=np.array([[c,0,s],[0,1,0],[-s,0,c]])
            qs.append([0.,math.sin(heading/2),0.,math.cos(heading/2)])
            ep.append(p+rot@offset);bp.append(p+rot@np.array([0.,max(.05,float(offset[1])*.8),0.]))
        rotations.append(qs);emitters.append(ep);bodies.append(bp)
    stacked = np.asarray(paths)
    ground_paths = [
        path for index, path in enumerate(paths)
        if _placement_is_ground(
            placement_rows.get(actors[index].get('entity_instance_id'))
        )
    ]
    if ground_paths:
        if contact_report and contact_report.get('per_trajectory_point'):
            floor_y = metadata.get('selected_floor_height_m')
        else:
            ok, floor_y = _points_same_floor(
                np.asarray(ground_paths),
                metadata.get('selected_floor_height_m'),
            )
            if not ok:
                raise CandidateFailure('routes', 'sources_not_on_same_floor')
    else:
        floor_y = metadata.get('selected_floor_height_m')
    metadata['selected_floor_height_m'] = floor_y
    metadata['selected_floor_reference_source'] = floor_source
    metadata['same_floor_tolerance_m'] = SAME_FLOOR_Y_TOLERANCE_M
    return stacked,np.asarray(rotations),np.asarray(moving),np.asarray(emitters),np.asarray(bodies),metadata


def _merge(ranges):
    result=[]
    for lo,hi in sorted((int(a),int(b)) for a,b in ranges if a<=b):
        if result and lo<=result[-1][1]+1: result[-1][1]=max(result[-1][1],hi)
        else: result.append([lo,hi])
    return result


def _clip_ranges(ranges,low=-math.inf,high=math.inf):
    return _merge((max(a,low),min(b,high)) for a,b in ranges if max(a,low)<=min(b,high))


def _intersect_ranges(first, second):
    """Intersect two sorted disjoint inclusive integer range lists."""
    result=[];i=j=0
    while i<len(first) and j<len(second):
        lo=max(first[i][0],second[j][0]);hi=min(first[i][1],second[j][1])
        if lo<=hi:result.append([lo,hi])
        if first[i][1]<second[j][1]:i+=1
        else:j+=1
    return result


def activity_intervals(sound):
    """Measured active intervals of one prepared clip, in that clip's samples.

    A prepared segment keeps the natural pauses of its recording, so the union
    of measured intervals is the emission, not the bounding span. A clip that
    declares no measurement keeps using its audible span, which is what every
    older pool row supplies.
    """
    start,end=int(sound['audible_start_sample']),int(sound['audible_end_sample_exclusive'])
    count=int(sound['sample_count'])
    if not 0<=start<end<=count:
        raise ValueError('prepared activity span is outside the clip')
    raw=sound.get('source_activity_intervals_samples')
    if not isinstance(raw,(list,tuple)) or not raw:
        return [(start,end)],'audible_span'
    intervals=[]
    for pair in raw:
        first,last=int(pair[0]),int(pair[1])
        if not 0<=first<last<=count:
            raise ValueError('prepared activity interval is outside the clip')
        intervals.append((first,last))
    intervals.sort()
    return intervals,'measured_activity_intervals'


def _pick_sample(ranges,rng):
    sizes=[b-a+1 for a,b in ranges]; total=sum(sizes)
    if total<=0: raise CandidateFailure('schedule','no_legal_integer_start_sample')
    draw=int(rng.integers(total))
    for (a,b),size in zip(ranges,sizes):
        if draw<size: return a+draw
        draw-=size
    raise AssertionError('unreachable integer range draw')


def legal_start_ranges(mask, sound, clock, profile):
    """Exact integer starts whose measured activity lies in legal frame runs.

    Every measured interval has to fall inside one legal run; a natural pause
    between two of them does not, because the source is not emitting then.
    """
    sr,fps=int(clock['sample_rate_hz']),float(clock['frame_rate_hz'])
    deadline=int(clock['sample_count'])-int(round(profile['reserve_tail_s']*sr))
    edges=np.diff(np.r_[False,np.asarray(mask,dtype=bool),False].astype(int))
    runs=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
    count=int(sound['sample_count'])
    intervals,_basis=activity_intervals(sound)
    result=None
    for a,b in intervals:
        ranges=_merge((max(0,int(math.ceil(start*sr/fps))-a),
                       min(deadline-count,int(math.floor(end*sr/fps))-b)) for start,end in runs)
        result=ranges if result is None else _intersect_ranges(result,ranges)
        if not result:
            return []
    return result or []


def anchor_event_subwindows(fov, sound, clock, start_sample):
    """Frames of one audible window, split into its visible and hidden parts.

    ``fov`` is the per-frame frustum series of the anchor for one camera pose.
    The window runs from the first measured activity sample to the last, so a
    natural pause inside a sounding segment stays inside the window.
    """
    sr, fps = int(clock['sample_rate_hz']), float(clock['frame_rate_hz'])
    intervals, _basis = activity_intervals(sound)
    if not intervals:
        return None
    first = int(start_sample) + int(intervals[0][0])
    last = int(start_sample) + int(intervals[-1][1])
    lo = max(0, int(math.floor(first * fps / sr)))
    hi = min(len(fov) - 1, int(math.ceil(last * fps / sr)) - 1)
    if hi < lo:
        return None
    series = np.asarray(fov, dtype=bool)[lo:hi + 1]
    frames = np.arange(lo, hi + 1)
    return {
        'window_frames': [int(lo), int(hi)],
        'visible_frames': [int(f) for f in frames[series]],
        'hidden_frames': [int(f) for f in frames[~series]],
    }


def _visible_then_hidden_ok(fov, sound, clock, start_sample):
    """Whether this placement can actually carry the AV question.

    Three things have to hold inside one audible window, not two: a visible
    frame that identifies the emitter, a later hidden stretch, and a whole
    second inside that hidden stretch for the question to point at. Dropping
    the third produced a plan whose hidden part was two frames long and had no
    publishable query instant in it at all - the shape was right and the
    question still could not be asked.
    """
    split = anchor_event_subwindows(fov, sound, clock, start_sample)
    if split is None or not split['visible_frames'] or not split['hidden_frames']:
        return False
    anchor = min(split['visible_frames'])
    hidden_after = [f for f in split['hidden_frames'] if f > anchor]
    if not hidden_after:
        return False
    fps = float(clock['frame_rate_hz'])
    first, last = min(hidden_after), max(hidden_after)
    # QA-25 publishes a whole-second query frame, so one has to exist here.
    return any(first <= second * fps <= last
               for second in range(int(math.ceil(first / fps)),
                                   int(math.floor(last / fps)) + 1))


def visible_then_hidden_start_ranges(ranges, fov, sound, clock):
    """Keep only the starts whose own audible window is visible then hidden.

    ``legal_start_ranges`` has already reduced the placements to those whose
    measured activity lies in legal frames. This is the extra statement QA-25 AV
    makes about the shape of that one window, so it is applied to the same
    ranges rather than to a second whole-clip mask. Candidates are stepped at
    frame granularity, which is the resolution the visibility series has.
    """
    if not ranges:
        return []
    sr, fps = int(clock['sample_rate_hz']), float(clock['frame_rate_hz'])
    step = max(1, int(round(sr / fps)))
    kept = []
    for low, high in ranges:
        low, high = int(low), int(high)
        run_start = None
        sample = low
        while sample <= high:
            if _visible_then_hidden_ok(fov, sound, clock, sample):
                if run_start is None:
                    run_start = sample
            elif run_start is not None:
                kept.append((run_start, sample - 1))
                run_start = None
            sample += step
        if run_start is None and _visible_then_hidden_ok(fov, sound, clock, high):
            kept.append((high, high))
        elif run_start is not None:
            kept.append((run_start, high))
    return _merge(kept) if kept else []


def _sequential_orders(events, starts, clock, profile):
    gap=int(round(profile['min_gap_between_audible_windows_s']*clock['sample_rate_hz']))
    originals=[key for key,event in events.items() if not event.get('repeat_of')]
    repeats=[key for key,event in events.items() if event.get('repeat_of')]
    for order in itertools.permutations(originals):
        orders=[order]
        if repeats:
            repeated=repeats[0]; source=events[repeated]['repeat_of']; idx=order.index(source)
            orders=[order[:pos]+(repeated,)+order[pos:] for pos in range(idx+1,len(order)+1)]
        for candidate in orders:
            latest={}; bound=math.inf; following=None
            for key in reversed(candidate):
                if following is not None:
                    bound=latest[following]+events[following]['audible_start_sample']-events[key]['audible_end_sample_exclusive']-gap
                legal=_clip_ranges(starts[key],high=bound)
                if not legal: break
                latest[key]=legal[-1][1]; following=key
            if len(latest)==len(candidate): yield candidate,latest


def _orders_with_first_speaker(orders, events, wanted):
    """Keep only the actor orders whose earliest original event is ``wanted``.

    ``_sequential_orders`` enumerates every feasible order and the caller used to
    draw one uniformly, so a question whose gold answer is the first speaker got
    whichever entity the draw happened to pick. Filtering here keeps the
    enumeration and its feasibility guarantees intact and only removes the orders
    that answer a different question.
    """
    kept=[]
    for order,latest in orders:
        original=[k for k in order if not events[k].get('repeat_of')]
        if not original:continue
        first=events[original[0]]
        if str(first.get('entity_instance_id') or first.get('actor_id'))==str(wanted):
            kept.append((order,latest))
    return kept


def schedule_legal_events(events, starts, clock, profile, rng=None):
    """Choose only starts with a feasible suffix; feasibility calls consume no RNG."""
    relation=profile['event_relation'];sr=int(clock['sample_rate_hz'])
    wanted_first=profile.get('first_speaker_instance_id')
    if relation in {'sequential','repeat'}:
        orders=list(_sequential_orders(events,starts,clock,profile))
        if wanted_first is not None:
            orders=_orders_with_first_speaker(orders,events,wanted_first)
        if not orders:return None
        if rng is None:return True
        # First select a feasible actor order uniformly, then legal repeat insertion.
        by_original={}
        for order,latest in orders:
            original=tuple(k for k in order if not events[k].get('repeat_of'))
            by_original.setdefault(original,[]).append((order,latest))
        groups=list(by_original.values());group=groups[int(rng.integers(len(groups)))];order,latest=group[int(rng.integers(len(group)))]
        result={}; previous=None;gap=int(round(profile['min_gap_between_audible_windows_s']*sr))
        for key in order:
            low=0 if previous is None else result[previous]+events[previous]['audible_end_sample_exclusive']+gap-events[key]['audible_start_sample']
            result[key]=_pick_sample(_clip_ranges(starts[key],low,latest[key]),rng);previous=key
        return result
    if wanted_first is not None:
        # Only a sequential schedule has a well-defined earliest speaker.
        raise ValueError(
            'first_speaker_instance_id requires event_relation=sequential or repeat, not '
            + str(relation))
    overlap=int(round(profile['minimum_overlap_s']*sr));pairs=[]
    for a,b in itertools.combinations(events,2):
        ea,eb=events[a],events[b]
        if ea['actor_id']==eb['actor_id'] or min(ea['audible_end_sample_exclusive']-ea['audible_start_sample'],eb['audible_end_sample_exclusive']-eb['audible_start_sample'])<overlap:
            continue
        low_delta=ea['audible_start_sample']+overlap-eb['audible_end_sample_exclusive']
        high_delta=ea['audible_end_sample_exclusive']-overlap-eb['audible_start_sample']
        legal=_merge((max(x,u-high_delta),min(y,v-low_delta)) for x,y in starts[a] for u,v in starts[b])
        if legal:pairs.append((a,b,legal,low_delta,high_delta))
    if not pairs or any(not ranges for ranges in starts.values()):return None
    if rng is None:return True
    a,b,legal,lo,hi=pairs[int(rng.integers(len(pairs)))];result={a:_pick_sample(legal,rng)}
    result[b]=_pick_sample(_clip_ranges(starts[b],result[a]+lo,result[a]+hi),rng)
    for key in events:
        if key not in result:result[key]=_pick_sample(starts[key],rng)
    return result


def _event_bindings(sounds, profile, rng):
    from avengine.dataset.source_capabilities import make_event_id
    result={f'event_{i+1:03d}':deepcopy(sounds[actor]) for i,actor in enumerate(sorted(sounds))}
    if profile['event_relation']=='repeat':
        keys=[k for k,e in result.items() if e.get('repeat_requested')];original=keys[0] if len(keys)==1 else list(result)[int(rng.integers(len(result)))];repeat=deepcopy(result[original]);repeat['repeat_of']=original
        result[f'event_{len(result)+1:03d}']=repeat
    # One instance may own several emissions; the ordinal is per instance, so a
    # second event of one entity never reads as a second entity.
    ordinals=Counter()
    for key in sorted(result):
        instance=result[key].get('entity_instance_id')
        if instance:
            ordinals[instance]+=1
            result[key]['instance_event_id']=make_event_id(str(instance),ordinals[instance])
            result[key]['event_ordinal_for_instance']=int(ordinals[instance])
    return result


def visibility_transition_ok(states, transition):
    """Whether one body-proxy frustum series crosses the way the question needs.

    This is a field-of-view crossing of the existing body proxy. A pixel
    occlusion state is a different measurement and is neither produced nor
    claimed here.
    """
    states=np.asarray(states,dtype=bool)
    inside=np.flatnonzero(states);outside=np.flatnonzero(~states)
    if transition=='none':
        return True
    if not len(inside) or not len(outside):
        return False
    if transition=='out_of_view_to_visible':
        return bool(outside.min()<inside.max())
    if transition=='visible_then_hidden':
        return bool(inside.min()<outside.max())
    raise ValueError('unsupported visibility_transition: '+str(transition))


def planned_query_window(events, clock, profile, request=None, anchor_actor_ids=None):
    """Per-event post-sound windows a question may ask about, with public bounds.

    A question that asks about what happened after one event needs a query
    instant later than that event and its wet tail, with no other event
    interfering, so each event's window ends where the next one starts rather
    than at the end of the clip. Planning cannot know the measured wet tail yet,
    so the window opens one reserved-tail budget after the planned audible end
    and the readback replaces that budget with the measurement. The public
    bounds are quantized inward exactly the way the catalog publishes them.
    """
    request=request or {}
    sr=int(clock['sample_rate_hz']);total=int(clock['sample_count'])
    reserve=float(profile.get('reserve_tail_s',0.))
    end_hold_samples=int(round(float(profile.get('end_hold_s',0.))*sr))
    precision=int(request.get('public_time_precision',profile.get('public_time_precision',0)))
    anchors=set(anchor_actor_ids or ())
    from avengine.qa.generation_conditions import integer_second_window
    ordered=sorted(events,key=lambda event:int(event['planned_audible_interval_samples'][0]))
    rows=[]
    for event in ordered:
        end=int(event['planned_audible_interval_samples'][1])
        others=[other for other in ordered if other['event_id']!=event['event_id']]
        # An event still sounding at this one's end pushes the boundary out; the
        # window then runs until the next event starts.
        boundary,changed=end,True
        while changed:
            changed=False
            for other in others:
                low,high=(int(value) for value in other['planned_audible_interval_samples'])
                if low<=boundary<high:
                    boundary,changed=high,True
        following=[int(other['planned_audible_interval_samples'][0]) for other in others
                   if int(other['planned_audible_interval_samples'][0])>boundary]
        limit=min(following) if following else total-end_hold_samples
        start_s=boundary/sr+reserve;end_s=limit/sr
        published=integer_second_window(start_s,end_s,precision=precision) if end_s>start_s else None
        rows.append({'event_id':event['event_id'],'actor_id':event.get('actor_id'),
                     'entity_instance_id':event.get('entity_instance_id'),
                     'is_anchor_event':event.get('actor_id') in anchors,
                     'window_s':[start_s,max(start_s,end_s)],
                     'available_s':max(0.,end_s-start_s),
                     'blocked_by_other_event':boundary!=end,
                     'ends_at_next_event':bool(following),
                     'public_query_window_s':list(published) if published else None})
    preferred=[row for row in rows if row['is_anchor_event']] or rows
    publishable=[row for row in preferred if row['public_query_window_s']]
    requires=bool(request.get('require_public_query_window',
                              profile.get('require_public_query_window',False)))
    return {'reserve_tail_s':reserve,'public_time_precision':precision,
            'windows':rows,
            'anchor_actor_ids':sorted(anchors),
            'public_query_window_s':publishable[0]['public_query_window_s'] if publishable else None,
            'publishable_event_ids':[row['event_id'] for row in publishable],
            'requires_public_window':requires,
            'window_authority':'derived_post_sound_window',
            'basis':('planned audible end plus the reserved tail budget, ending where the '
                     'next event starts; the measured wet tail replaces that budget at readback'),
            'wet_tail_status':'not_run'}


def _visibility_requirements_for_selection(
    compiled_conditions, actors, profile, frame_count
):
    """Translate the selected compiled target into P04 pixel requirements.

    The compiler may emit one candidate per speaking instance. The sampler has
    already chosen its anchor role, so only that target's visibility recipe
    may constrain this camera; otherwise two alternative answer candidates
    would be required to hold simultaneously.
    """
    if not compiled_conditions:
        return ()
    from avengine.rooms import conditioned_visibility as cv

    anchor_ids = {
        str(actors[index]["entity_instance_id"])
        for index in profile["anchor_indices"]
    }

    def target_id(compiled):
        subjects = (
            getattr(compiled, "subjects", None)
            if not isinstance(compiled, Mapping)
            else compiled.get("subjects")
        )
        for subject in subjects or ():
            if isinstance(subject, Mapping):
                role = subject.get("role")
                value = subject.get("entity_instance_id")
            else:
                role = getattr(subject, "role", None)
                value = getattr(subject, "entity_instance_id", None)
            if role == "target" and value is not None:
                return str(value)
        return None

    selected = [
        item for item in compiled_conditions if target_id(item) in anchor_ids
    ] or list(compiled_conditions)
    windows = {
        str(actor["entity_instance_id"]): [[0, int(frame_count)]]
        for actor in actors
    }
    precision = int(profile.get("public_time_precision", 0))
    requirements = []
    for compiled in selected:
        payload = (
            compiled.to_dict()
            if hasattr(compiled, "to_dict")
            else dict(compiled)
        )
        requirements.extend(
            cv.requirements_from_conditions(
                payload,
                observation_windows_by_subject=windows,
                public_time_precision=precision,
                qa_id=payload.get("qa_id"),
            )
        )
    unique = []
    seen = set()
    for requirement in requirements:
        identity = (
            requirement.kind,
            requirement.subject,
            requirement.state,
            requirement.side,
            requirement.require_complete_coverage,
            requirement.require_publishable_window,
        )
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(requirement)
    return tuple(unique)


def _visibility_solver_options(request, frame_count):
    camera = request.get("camera", {})
    if not isinstance(camera, Mapping):
        camera = {}
    options = camera.get("visibility_solver")
    if options is None:
        options = camera.get("visibility", {})
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise CandidateFailure("camera", "visibility_solver_config_not_mapping")

    def positive_integer(name, default):
        value = options.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise CandidateFailure(
                "camera", "visibility_solver_%s_not_positive_integer" % name
            )
        value = int(value)
        if value < 1:
            raise CandidateFailure(
                "camera", "visibility_solver_%s_not_positive_integer" % name
            )
        return value

    frame_budget = positive_integer("frame_budget", int(frame_count))
    if frame_budget < int(frame_count):
        raise CandidateFailure(
            "camera", "visibility_frame_budget_below_required_clock"
        )
    max_candidates = positive_integer("max_candidates", 8)
    max_ray_poses = positive_integer("max_ray_poses", 16)
    candidate_pool_budget = positive_integer(
        "candidate_pool_budget",
        max(64, max_candidates * 8, max_ray_poses * 8),
    )
    body_geometry = options.get("body_geometry", "emitter_proxy")
    if body_geometry not in {"emitter_proxy", "registered_body_envelope"}:
        raise CandidateFailure("camera", "unsupported_visibility_body_geometry")
    return {
        "body_geometry": body_geometry,
        "max_candidates": max_candidates,
        "max_ray_poses": max_ray_poses,
        "candidate_pool_budget": candidate_pool_budget,
        "frame_budget": frame_budget,
        "screen_policy": options.get("screen_policy", "validated_3x3_v1"),
    }


def _native_visibility_not_run():
    from avengine.rooms.conditioned_visibility import accept_native_visibility

    return {
        "status": "not_run",
        "tier": "native_pixel",
        "authority": "native_pixel_witness",
        "entrypoint": (
            f"{accept_native_visibility.__module__}."
            f"{accept_native_visibility.__name__}"
        ),
        "reason": "native visual/RLR budget is 0; no pixel truth was supplied",
    }


def select_camera_and_schedule(space, mesh, paths, moving, emitters, bodies, actors,
                                sounds, profile, clock, request, rng, region=None, *,
                                event_bindings=None, fixed_schedule=None,
                                motion_requirements=None, motion_solution=None,
                                visibility_requirements=(), room_package=None,
                                static_placements=None, root_rotations=None,
                                source_registry=None):
    config=request.get('camera',{});fov=float(config.get('fov_deg',request.get('camera_fov_deg',85.)))
    height=float(config.get('height_above_floor_m',1.55))
    resolution=list(config.get('resolution_hw',[720,1280]))
    if len(resolution)!=2 or any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in resolution):raise ValueError('camera resolution must be positive integer [height,width]')
    aspect=resolution[1]/resolution[0]
    if not 0<fov<180 or height<=0:raise ValueError('invalid static camera FOV/height')
    placement_rows = _static_placement_rows(static_placements)
    ground_paths = [
        path for index, path in enumerate(paths)
        if _placement_is_ground(
            placement_rows.get(actors[index].get('entity_instance_id'))
        )
    ]
    if ground_paths:
        floor_ok, floor_y = _points_same_floor(
            np.asarray(ground_paths)
        )
        floor_source = 'ground_actor_paths'
    else:
        floor_y, floor_source = static_support_floor_reference(
            room_package or {}, space, placement_rows
        )
        floor_ok = True
    if not floor_ok:
        raise CandidateFailure('camera', 'sources_not_on_same_floor')
    cam_region = space.bounds().copy() if region is None else np.asarray(region, dtype=float).copy()
    cam_region[0, 1] = floor_y - SAME_FLOOR_Y_TOLERANCE_M
    cam_region[1, 1] = floor_y + SAME_FLOOR_Y_TOLERANCE_M
    positions=_camera_grid_on_floor(space, height=height, region=cam_region, floor_y=floor_y)
    yaws=np.deg2rad(np.arange(0,360,15)); forwards=np.c_[np.sin(yaws),np.zeros(24),-np.cos(yaws)]
    rights=np.c_[np.cos(yaws),np.zeros(24),np.sin(yaws)];tangent=math.tan(math.radians(fov)/2)
    events=event_bindings if event_bindings is not None else _event_bindings(sounds,profile,rng)
    actor_index={a['actor_id']:i for i,a in enumerate(actors)}
    anchors=set(profile['anchor_indices']); legal=[]; stages=Counter();ray_cache={}
    def fixed_schedule_is_legal(starts):
        if fixed_schedule is None:
            return bool(starts) and all(starts.values()) and (
                schedule_legal_events(events, starts, clock, profile) is not None)
        if set(fixed_schedule) != set(events):
            return False
        for key, ranges in starts.items():
            value = fixed_schedule.get(key)
            if not isinstance(value, (int, np.integer)):
                return False
            if not any(int(low) <= int(value) <= int(high) for low, high in ranges):
                return False
        singleton = {
            key: [[int(fixed_schedule[key]), int(fixed_schedule[key])]]
            for key in events
        }
        return schedule_legal_events(events, singleton, clock, profile) is not None
    distance_range=profile.get('distance_range_m') or request.get('profile',{}).get('distance_range_m') or DEFAULT_DISTANCE_RANGE_M
    distance_range=[float(distance_range[0]), float(distance_range[1])]
    competitor_visibility=profile.get('competitor_visibility', 'in_fov')
    competitor_motion=profile.get('competitor_motion', COMPETITOR_MOTION_DEFAULT)
    transition=profile.get('visibility_transition', VISIBILITY_TRANSITION_DEFAULT)
    competitors=[j for j in range(len(actors)) if j not in anchors]
    visibility_report = {
        'status': 'not_requested',
        'tier': 'not_run',
        'requirements': [],
        'candidate_ids_applied': [],
        'claim_boundary': (
            'no compiled pixel visibility requirement reached camera selection'
        ),
    }
    visibility_allowed_ids = None
    visibility_native_acceptance = _native_visibility_not_run()
    visibility_tracks = None
    visibility_options = None
    visibility_geometry_authority = "unknown"
    if visibility_requirements:
        from avengine.rooms import conditioned_visibility as cv

        visibility_options = _visibility_solver_options(
            request, int(clock["frame_count"])
        )
        visibility_tracks = []
        envelope_subjects = {requirement.subject for requirement in visibility_requirements}
        envelope_subjects.update(requirement.occluder_subject for requirement in visibility_requirements
                                 if requirement.occluder_subject)
        asset_records = {str(row["asset_id"]): row
                         for row in (source_registry or {}).get("assets", ())}
        body_geometry_evidence = {}
        for index, actor in enumerate(actors):
            binding = actor.get("emitter_binding") or {}
            emitter_offset = binding.get("emitter_offset_m")
            if emitter_offset is None:
                emitter_offset = actor.get("emitter_offset_m")
            envelope = None
            instance_id = str(actor["entity_instance_id"])
            if (visibility_options["body_geometry"] == "registered_body_envelope"
                    and instance_id in envelope_subjects):
                from avengine.assets.qualification_geometry import measure_registered_body_envelope
                record = asset_records.get(str(actor["asset_id"]))
                if record is None or root_rotations is None:
                    raise CandidateFailure("camera", "registered_body_geometry_input_missing")
                measured = measure_registered_body_envelope(record)
                if measured.get("measurement") != "measured":
                    raise CandidateFailure("camera", "registered_body_geometry_unavailable:" + instance_id)
                local_points = np.asarray(measured["vertices_m"], dtype=float)
                world_points = np.stack([
                    local_points @ _rotation_matrix_from_xyzw(rotation).T + point
                    for rotation, point in zip(root_rotations[index], paths[index])
                ])
                envelope = cv.ActorBodyEnvelope(
                    vertices_m=world_points, source_ref=measured["source_ref"],
                    pose_coverage=measured["pose_coverage"])
                body_geometry_evidence[instance_id] = {
                    key: deepcopy(measured[key]) for key in
                    ("asset_id", "source_ref", "frame", "bounds_min_m", "bounds_max_m",
                     "pose_coverage", "action_frame_counts")
                }
            visibility_tracks.append(
                cv.ActorTrack(
                    instance_id=str(actor["entity_instance_id"]),
                    positions_m=np.asarray(paths[index], dtype=float),
                    body=cv.body_proxy_from_emitter_anchor(emitter_offset),
                    entity_class=str(actor["entity_class"]),
                    body_envelope=envelope,
                )
            )
        visibility_geometry_authority = cv.geometry_authority_for_package(
            room_package
        )

    trend = (
        dict(getattr(motion_solution, 'distance_trend', {}))
        if motion_solution is not None else {}
    )
    trend_target_index = None
    trend_frames = None
    if trend.get('expected_sign') is not None and motion_solution is not None:
        target_requirement = next(
            (item for item in motion_solution.requirements if item.role == 'target'),
            None,
        )
        target_id = (
            target_requirement.entity_instance_id
            if target_requirement is not None else None
        )
        trend_target_index = next(
            (index for index, actor in enumerate(actors)
             if actor.get('entity_instance_id') == target_id),
            None,
        )
        trend_frames = trend.get('judged_over_frames')
        if trend_frames is None and target_requirement is not None:
            trend_frames = target_requirement.moving_frames
        if trend_target_index is None or trend_frames is None:
            raise CandidateFailure(
                'camera', 'motion_solver_distance_trend_target_unavailable')
    cheap_legal = []
    position_states = {}
    for pi,position in enumerate(positions):
        origin=np.asarray(position)
        floor=origin-np.array([0,height,0])
        if any(np.linalg.norm(path-floor,axis=1).min()<.8 for path in paths):
            continue
        stages['camera_positions_after_clearance']+=1
        delta=bodies-origin
        depth=np.einsum('yc,nfc->ynf',forwards,delta)
        side=np.einsum('yc,nfc->ynf',rights,delta)
        fov_mask=(depth>.1)&(np.abs(side)<depth*tangent*.93)&(
            np.abs(delta[None,:,:,1])<depth*tangent/aspect*.9)
        # The entry/exit solver and the event mask must screen the same body.
        # An emitter-point inset can exclude a partly visible body, defeating a
        # valid edge-entry scene. Envelope overlap is only a candidate filter;
        # native pixels remain the acceptance authority.
        if visibility_tracks is not None:
            for body_index, track in enumerate(visibility_tracks):
                if track.body_envelope is None:
                    continue
                points_delta = track.body_envelope.vertices_m - origin
                full_depth = np.einsum('yc,fvc->yfv', forwards, points_delta)
                full_side = np.einsum('yc,fvc->yfv', rights, points_delta)
                ahead = full_depth > cv.screen_policy(visibility_options['screen_policy']).near_m
                safe_depth = np.where(ahead, full_depth, np.nan)
                columns = (resolution[1]-1)/2 + full_side/safe_depth/tangent*resolution[1]/2
                rows = (resolution[0]-1)/2 - points_delta[None,:,:,1]/safe_depth/(tangent/aspect)*resolution[0]/2
                valid = ahead & np.isfinite(columns) & np.isfinite(rows)
                projected_overlap = (
                    (np.max(np.where(valid,columns,-np.inf),axis=-1) >= -.5)
                    & (np.min(np.where(valid,columns,np.inf),axis=-1) <= resolution[1]-.5)
                    & (np.max(np.where(valid,rows,-np.inf),axis=-1) >= -.5)
                    & (np.min(np.where(valid,rows,np.inf),axis=-1) <= resolution[0]-.5))
                fov_mask[:,body_index] = projected_overlap
        d=emitters-origin
        distance=np.linalg.norm(d,axis=-1)
        in_range=(distance>=distance_range[0])&(distance<=distance_range[1])
        az=np.degrees(np.arctan2(d[:,:,0],-d[:,:,2]))
        separation=np.abs((az[:,None]-az[None,:]+180)%360-180)
        for i in range(len(actors)):
            separation[i,i]=np.inf
        nearest=separation.min(axis=1)
        mask=np.ones_like(fov_mask,dtype=bool)
        if trend_target_index is not None:
            from avengine.rooms.conditioned_motion import distance_trend
            trend_report = distance_trend(
                emitters[trend_target_index], position, trend_frames, criterion=trend)
            if (
                trend_report['sign'] != trend['expected_sign']
                or not trend_report['meets_margin']
                or not trend_report['monotone_within_tolerance']
            ):
                stages['poses_without_requested_distance_trend'] += 1
                continue
        low,high=profile['separation_bin_deg']
        if transition!='none':
            keep=np.array([all(visibility_transition_ok(
                fov_mask[yi,i],transition)
                for i in profile['anchor_indices']) for yi in range(24)])
            if not keep.any():
                stages['poses_without_requested_visibility_transition']+=1
            mask[~keep]=False
        if competitor_visibility=='off_screen':
            silent_in_fov=np.zeros(24, dtype=bool)
            for i in range(len(actors)):
                if i not in anchors:
                    silent_in_fov |= np.any(fov_mask[:, i], axis=1)
            mask[silent_in_fov]=False
        anchor_visibility=profile['anchor_visibility']
        for i in profile['speaking_indices']:
            if i in anchors:
                if anchor_visibility=='visible_then_hidden':
                    # Both states are legal inside the window; which frames may
                    # carry which is decided per placement below, because the
                    # statement is about the order inside one event.
                    visible=np.ones_like(fov_mask[:,i],dtype=bool)
                else:
                    visible=fov_mask[:,i] if anchor_visibility=='in_fov' else ~fov_mask[:,i]
            else:
                visible=fov_mask[:,i] if competitor_visibility=='in_fov' else ~fov_mask[:,i]
            mask[:,i]&=visible&in_range[i][None]
            if i in anchors:
                mask[:,i]&=(nearest[i]>=low)[None]&(
                    (nearest[i]<high)
                    | ((high==180)&np.isclose(nearest[i],180))
                )[None]
                if motion_requirements is None:
                    if profile['speech_motion']=='speaker_moving':
                        mask[:,i]&=moving[i][None]
                    elif profile['speech_motion']=='competitor_moving':
                        mask[:,i]&=np.any(
                            np.delete(moving,i,axis=0),axis=0
                        )[None]
                    elif profile['speech_motion']=='all_still':
                        # all_still is the *unstated* default of speech_motion,
                        # so it is a silent positive claim, not the question's.
                        # An instance the visibility route plan had to move is
                        # exempt from it; a stated or compiled all_still is not.
                        forced=set(profile.get('visibility_forced_movers') or ())
                        stated=(profile.get('knob_sources') or {}).get(
                            'speech_motion') is not None
                        still_rows=[
                            index for index in range(len(actors))
                            if stated or str(
                                actors[index]['entity_instance_id']) not in forced
                        ]
                        if competitor_motion!=COMPETITOR_MOTION_DEFAULT:
                            if i in still_rows:
                                mask[:,i]&=~moving[i][None]
                        elif still_rows:
                            mask[:,i]&=~np.any(moving[still_rows],axis=0)[None]
                    if competitors:
                        if competitor_motion=='moving':
                            mask[:,i]&=np.all(
                                moving[competitors],axis=0
                            )[None]
                        elif competitor_motion=='still':
                            mask[:,i]&=~np.any(
                                moving[competitors],axis=0
                            )[None]
        if motion_requirements:
            for i, actor in enumerate(actors):
                requirement = motion_requirements.get(
                    actor.get('entity_instance_id')
                )
                if requirement is None:
                    continue
                required_motion = np.ones(paths.shape[1], dtype=bool)
                for first, last in requirement.still_frames:
                    required_motion[int(first):int(last)] &= ~moving[
                        i, int(first):int(last)
                    ]
                if requirement.moving_frames is not None:
                    first, last = map(int, requirement.moving_frames)
                    required_motion[first:last] &= moving[i, first:last]
                mask[:, i] &= required_motion[None]
        before=[]
        starts_by_yaw={}
        for yi in range(24):
            starts={
                key:legal_start_ranges(
                    mask[yi,actor_index[e['actor_id']]],
                    e,clock,profile
                )
                for key,e in events.items()
            }
            if profile['anchor_visibility']=='visible_then_hidden':
                for key,e in events.items():
                    index=actor_index[e['actor_id']]
                    if index not in anchors:
                        continue
                    starts[key]=visible_then_hidden_start_ranges(
                        starts[key],fov_mask[yi,index],e,clock)
                if any(not starts[key] for key,e in events.items()
                       if actor_index[e['actor_id']] in anchors):
                    stages['poses_without_visible_then_hidden_anchor_event']+=1
                    continue
            if fixed_schedule_is_legal(starts):
                before.append(yi)
                starts_by_yaw[yi]=starts
        if not before:
            continue
        stages['poses_with_projection_angle_motion_schedule']+=len(before)
        position_states[pi] = {
            'origin': origin,
            'mask': mask.copy(),
        }
        cheap_legal.extend(
            (pi,yi,starts if isinstance(starts, Mapping) else {},nearest.copy())
            for yi in before
        )

    stages['cheap_candidate_count'] = len(cheap_legal)
    if not cheap_legal:
        raise CandidateFailure('camera', 'no_joint_geometry_activity_schedule')
    visibility_selection = {
        'mode': 'all_cheap_candidates',
        'selection_policy': 'none',
        'candidate_pool_budget': None,
        'candidate_pool_count': len(cheap_legal),
        'candidate_pool_ids': [
            f'grid_{pi:05d}_yaw_{yi * 15:03d}'
            for pi,yi,_starts,_nearest in cheap_legal
        ],
        'projection_viable_ids': [],
        'ray_pose_ids': [],
        'ray_pose_count': len(cheap_legal),
    }
    ray_rows = cheap_legal
    if visibility_requirements:
        def _visibility_pose(position_index, yaw_index):
            return cv.CameraPose(
                candidate_id=(
                    f'grid_{position_index:05d}_yaw_{yaw_index * 15:03d}'
                ),
                position_m=tuple(
                    float(value) for value in positions[position_index]
                ),
                forward=tuple(
                    float(value) for value in forwards[yaw_index]
                ),
                right=tuple(float(value) for value in rights[yaw_index]),
                up=(0.0, 1.0, 0.0),
                horizontal_fov_deg=fov,
                resolution_hw=tuple(resolution),
            )

        order=np.arange(len(cheap_legal),dtype=int)
        rng.shuffle(order)
        pool_rows=[
            cheap_legal[int(index)]
            for index in order[:visibility_options['candidate_pool_budget']]
        ]
        projection_poses=[
            _visibility_pose(pi,yi)
            for pi,yi,_starts,_nearest in pool_rows
        ]
        projection_report = cv.solve_visibility_candidates(
            visibility_requirements,
            camera_poses=projection_poses,
            tracks=visibility_tracks,
            frame_rate_hz=float(clock['frame_rate_hz']),
            mesh=None,
            policy=visibility_options['screen_policy'],
            geometry_authority=visibility_geometry_authority,
            public_time_precision=int(profile.get('public_time_precision', 0)),
            max_candidates=len(projection_poses),
            max_ray_poses=len(projection_poses),
            frame_budget=visibility_options['frame_budget'],
        )
        projection_viable_ids={
            str(candidate['candidate_id'])
            for candidate in projection_report.get('candidates', ())
        }
        pool_ids=[
            f'grid_{pi:05d}_yaw_{yi * 15:03d}'
            for pi,yi,_starts,_nearest in pool_rows
        ]
        viable_in_selection_order=[
            row for row in pool_rows
            if f'grid_{row[0]:05d}_yaw_{row[1] * 15:03d}'
            in projection_viable_ids
        ]
        ray_rows=viable_in_selection_order[
            :visibility_options['max_ray_poses']
        ]
        visibility_selection = {
            'mode': 'bounded_projection_then_ray',
            'selection_policy': 'request_rng_permutation_v1',
            'candidate_pool_budget': visibility_options['candidate_pool_budget'],
            'candidate_pool_count': len(pool_rows),
            'candidate_pool_ids': pool_ids,
            'projection_viable_ids': [
                f'grid_{pi:05d}_yaw_{yi * 15:03d}'
                for pi,yi,_starts,_nearest in viable_in_selection_order
            ],
            'ray_pose_ids': [
                f'grid_{pi:05d}_yaw_{yi * 15:03d}'
                for pi,yi,_starts,_nearest in ray_rows
            ],
            'ray_pose_count': len(ray_rows),
            'projection_screen': {
                'status': projection_report.get('status'),
                'tier': projection_report.get('tier'),
                'camera_pose_count': projection_report.get(
                    'camera_pose_count'
                ),
                'candidate_count': len(
                    projection_report.get('candidates', ())
                ),
                'refuted_count': len(projection_report.get('refuted', ())),
                'stages': projection_report.get('stages'),
                'frame_budget': projection_report.get('budgets'),
            },
        }
        stages['cheap_candidate_pool_count'] = len(pool_rows)
        stages['cheap_projection_viable_count'] = len(
            viable_in_selection_order
        )
        stages['bounded_ray_candidate_count'] = len(ray_rows)
        if not ray_rows:
            visibility_report = projection_report
            visibility_report['bounded_selection'] = visibility_selection
            visibility_report['body_geometry_evidence'] = body_geometry_evidence
            visibility_report['native_acceptance'] = deepcopy(
                visibility_native_acceptance
            )
            raise CandidateFailure(
                'camera', 'visibility_solver_candidate_budget_exhausted'
            )

    selected_by_position = {}
    for row in ray_rows:
        selected_by_position.setdefault(int(row[0]), []).append(row)
    for pi, rows_for_position in selected_by_position.items():
        state = position_states[pi]
        mask = state['mask']
        selected_yaws = sorted({int(row[1]) for row in rows_for_position})
        needed=np.any(mask[selected_yaws],axis=0)
        origin=state['origin']
        stages['ray_frame_evaluations'] += sum(
            int(np.count_nonzero(needed[index]))
            for index in profile['speaking_indices']
        )
        los=np.zeros((len(actors),paths.shape[1]),dtype=bool)
        for i in profile['speaking_indices']:
            desired='blocked' if (
                i in anchors and profile['anchor_line_of_sight']=='occluded'
            ) else 'clear'
            for frame in np.flatnonzero(needed[i]):
                dest=emitters[i,frame]
                key=(pi,tuple(dest))
                if key not in ray_cache:
                    stages['ray_los_queries'] += 1
                    ray_cache[key]=line_of_sight(mesh,origin,dest)
                los[i,frame]=ray_cache[key]==desired
                if los[i,frame] and desired=='clear':
                    body_key=(pi,tuple(bodies[i,frame]))
                    if body_key not in ray_cache:
                        stages['ray_los_queries'] += 1
                        ray_cache[body_key]=line_of_sight(
                            mesh,origin,bodies[i,frame]
                        )
                    los[i,frame]=ray_cache[body_key]=='clear'
        mask_with_los=mask.copy()
        mask_with_los&=los[None]
        stages['ray_camera_positions'] += 1
        stages['ray_poses_evaluated'] += len(rows_for_position)
        for pi_row,yi,_starts,_nearest in rows_for_position:
            starts={
                key:legal_start_ranges(
                    mask_with_los[yi,actor_index[e['actor_id']]],
                    e,clock,profile
                )
                for key,e in events.items()
            }
            if fixed_schedule_is_legal(starts):
                legal.append((pi_row,yi,starts,_nearest.copy()))
            else:
                stages['ray_pose_schedule_rejections'] += 1

    stages['bounded_los_frame_count'] = int(
        stages.get('ray_frame_evaluations', 0)
    )
    visibility_selection['ray_frame_evaluations'] = stages.get(
        'ray_frame_evaluations', 0
    )
    visibility_selection['ray_los_queries'] = stages.get(
        'ray_los_queries', 0
    )
    if not legal:
        reason=(
            'visibility_solver_candidate_budget_exhausted'
            if visibility_requirements
            else (
                'no_joint_geometry_activity_schedule'
                if mesh is not None
                else 'static_geometry_unmeasured'
            )
        )
        raise CandidateFailure('camera',reason)
    if visibility_requirements:
        visibility_poses = [
            _visibility_pose(position_index, yaw_index)
            for position_index,yaw_index,_starts,_nearest in legal
        ]
        visibility_report = cv.solve_visibility_candidates(
            visibility_requirements,
            camera_poses=visibility_poses,
            tracks=visibility_tracks,
            frame_rate_hz=float(clock['frame_rate_hz']),
            mesh=mesh,
            policy=visibility_options['screen_policy'],
            geometry_authority=visibility_geometry_authority,
            public_time_precision=int(profile.get('public_time_precision', 0)),
            max_candidates=visibility_options['max_candidates'],
            max_ray_poses=visibility_options['max_ray_poses'],
            frame_budget=visibility_options['frame_budget'],
        )
        visibility_report['bounded_selection'] = visibility_selection
        visibility_report['body_geometry_evidence'] = body_geometry_evidence
        visibility_report['native_acceptance'] = deepcopy(
            visibility_native_acceptance
        )
        visibility_allowed_ids = {
            str(candidate['candidate_id'])
            for candidate in visibility_report.get('candidates', ())
        }
        # Prefer candidates that satisfy the screen over unresolved candidates.
        # Keep the unresolved route available when the screen cannot confirm
        # any candidate; only native pixels can establish final acceptance.
        consistent_ids = {
            str(candidate['candidate_id'])
            for candidate in visibility_report.get('candidates', ())
            if candidate.get('verdict') == 'consistent'
        }
        if consistent_ids:
            visibility_allowed_ids = consistent_ids
        visibility_report['selection_preference'] = (
            'screen_consistent' if consistent_ids else 'screen_unresolved')
        visibility_report['candidate_ids_applied'] = sorted(visibility_allowed_ids)
        legal = [
            row for row in legal
            if f'grid_{row[0]:05d}_yaw_{row[1] * 15:03d}'
            in visibility_allowed_ids
        ]
        if not legal:
            raise CandidateFailure(
                'camera', 'visibility_solver_candidate_budget_exhausted'
            )
    def _anchor_sep(nearest_deg):
        seps=[]
        for i in profile['anchor_indices']:
            vals=np.asarray(nearest_deg[i], dtype=float)
            finite=vals[np.isfinite(vals)]
            if len(finite):
                seps.append(float(np.median(finite)))
        return float(np.median(seps)) if seps else float('nan')
    sep_policy=profile.get('separation_target_policy', SEPARATION_TARGET_ANY_LEGAL)
    target_deg=None
    if sep_policy==SEPARATION_TARGET_UNIFORM_IN_BIN:
        target_deg=float(rng.uniform(float(low), float(high)))
        matched=[row for row in legal if abs(_anchor_sep(row[3])-target_deg)<=2.0]
        if not matched:
            raise CandidateFailure('camera', 'no_pose_within_uniform_in_bin_tolerance')
        legal=matched
    pi,yi,starts,nearest=legal[int(rng.integers(len(legal)))]
    schedule=(dict(fixed_schedule) if fixed_schedule is not None
              else schedule_legal_events(events,starts,clock,profile,rng))
    if not isinstance(schedule, Mapping):
        raise CandidateFailure('schedule', 'selected_camera_lost_fixed_event_schedule')
    achieved_sep=_anchor_sep(nearest)
    camera={'candidate_id':f'grid_{pi:05d}_yaw_{yi*15:03d}','position_m':positions[pi],
            'basis':{'forward':forwards[yi].tolist(),'right':rights[yi].tolist(),'up':[0.,1.,0.]},
            'horizontal_fov_deg':fov,'resolution_hw':resolution,'height_above_floor_m':height,'motion':'static','yaw_deg':int(yi*15)}
    sr=int(clock['sample_rate_hz']);tb=int(clock['time_base_hz']);render=request.get('audio_render',{});output=[]
    for key,event in events.items():
        start=int(schedule[key]);end=start+int(event['sample_count']);out=deepcopy(event)
        out.update(event_id=key,start_sample=start,end_sample=end,start_tick=int(round(start*tb/sr)),end_tick=int(round(end*tb/sr)),
                   linear_gain=float(render.get('linear_gain',event.get('linear_gain',1.))),
                   event_unit='independent_source_playback_onset',
                   planned_audible_interval_samples=[start+int(event['audible_start_sample']),start+int(event['audible_end_sample_exclusive'])])
        output.append(out)
    output.sort(key=lambda e:(e['start_sample'],e['event_id']))
    query=planned_query_window(output,clock,profile,request,
                               anchor_actor_ids=[actors[i]['actor_id'] for i in profile['anchor_indices']])
    if query['requires_public_window'] and query['public_query_window_s'] is None:
        # Bounded refusal: the remaining window cannot state one whole display
        # unit, which is a planning fact, not a rendering accident.
        raise CandidateFailure('question','integer_query_window_below_one_display_unit')
    return camera,output,{'selection':(visibility_selection['mode'] if visibility_requirements else 'uniform_over_all_legal_static_poses'),
                         'selection_policy':visibility_selection['selection_policy'],
                         'bounded_candidate_pool_count':visibility_selection['candidate_pool_count'],
                         'bounded_ray_pose_count':visibility_selection['ray_pose_count'],
                         'legal_candidate_count':len(legal),
                         'candidate_position_count':len(positions),'yaw_candidate_count':24,'stages':dict(stages),
                         'legal_candidate_ids':[f'grid_{p:05d}_yaw_{y*15:03d}' for p,y,_,_ in legal],
                         'legal_event_start_ranges_samples':starts,'nearest_competitor_separation_deg':nearest.tolist(),
                         'line_of_sight_source':deepcopy(getattr(mesh,'source',None)),
                         'anchor_indices':profile['anchor_indices'],'clear_los_requires':['emitter','body_proxy'],'pixel_observability':'not_run',
                         'requested_separation_bin_deg': list(map(float, profile['separation_bin_deg'])),
                         'planned_anchor_nearest_competitor_separation_deg': achieved_sep,
                         'separation_target_policy': sep_policy, 'separation_target_deg': target_deg,
                         'distance_trend_criterion_source': trend.get('criterion_source'),
                         'planned_distance_trend': (
                             None if trend_target_index is None else
                             __import__('avengine.rooms.conditioned_motion',
                                        fromlist=['distance_trend']).distance_trend(
                                            emitters[trend_target_index], positions[pi],
                                            trend_frames, criterion=trend)
                         ),
                         'separation_coverage_unit': 'achieved_angle_5deg_bins',
                         'planned_separation_histogram_5deg': histogram_separation_5deg([achieved_sep]),
                         'selected_floor_height_m': floor_y,
                         'selected_floor_reference_source': floor_source,
                         'same_floor_tolerance_m': SAME_FLOOR_Y_TOLERANCE_M,
                         'distance_range_m': list(distance_range),
                         'visibility_solver': visibility_report,
                         'visibility_native_acceptance': visibility_native_acceptance,
                         'anchor_visibility': profile.get('anchor_visibility'),
                         'competitor_visibility': competitor_visibility,
                         'competitor_motion': competitor_motion,
                         'visibility_transition': transition,
                         'visibility_transition_basis': 'body_proxy_field_of_view_series',
                         'planned_query_window': query,
                         'clip_span_fit_policy': CLIP_SPAN_FIT_POLICY}


def instance_rows(profile, request, actors=None, sounds=None, moving=None):
    """Describe the physical instances this Episode plans, for the compiler and facts.

    Before selection the rows carry the declared slot identity, so a request that
    names its targets compiles before an asset is drawn. After selection they
    carry the real instance IDs, so the same statement is recompiled against what
    was actually chosen.
    """
    declared=profile.get('instances') or []
    explicit=request.get('source_asset_ids') if isinstance(request,Mapping) else None
    anchors=set(profile['anchor_indices']);speaking=set(profile['speaking_indices'])
    rows=[]
    for i,source_class in enumerate(profile['source_classes']):
        row=declared[i] if i<len(declared) else {}
        slot=str((row or {}).get('source_slot_id') or f'source{i+1}')
        if actors is not None:
            actor=actors[i]
            entry={'entity_instance_id':actor['entity_instance_id'],'actor_id':actor['actor_id'],
                   'source_endpoint_id':actor['source_endpoint_id'],'asset_id':actor['asset_id'],
                   'asset_revision':actor['asset_revision'],'entity_class':actor['entity_class'],
                   'instance_ordinal':actor['instance_ordinal']}
        else:
            entry={'entity_instance_id':str((row or {}).get('entity_instance_id') or slot),
                   'actor_id':slot,'source_endpoint_id':f'{slot}_mouth',
                   'asset_id':(explicit[i] if explicit is not None else (row or {}).get('asset_id')),
                   'asset_revision':None,'entity_class':None,'instance_ordinal':None}
        entry.update({'source_slot_id':slot,'source_class':source_class,
                      'role':'target' if i in anchors else 'competitor',
                      'is_anchor':i in anchors,'speaking':i in speaking})
        selected=(sounds or {}).get(i) if isinstance(sounds,Mapping) else None
        if selected:
            entry['sound_asset_id']=selected.get('sound_asset_id')
            entry['sound_identity_id']=selected.get('sound_identity_id')
            entry['sound_class']=selected.get('sound_class')
        if moving is not None:
            entry['planned_moving_frames']=int(np.count_nonzero(moving[i]))
        rows.append(entry)
    return rows


def _question_targets_drive_sampler(request):
    targets = request.get('qa_targets') if isinstance(request, Mapping) else None
    branches = request.get('question_branches') if isinstance(request, Mapping) else None
    if branches:
        return True
    if request.get('drive_sampler_from_questions'):
        return True
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        return False
    if not targets:
        return False
    # ProductionSpec emits derived targets for a bare qa_ids list. They keep
    # compile/readback visibility but must not turn every default branch into
    # one sampler recipe. Explicit/config targets retain driving authority.
    derived = all(
        isinstance(target, Mapping)
        and target.get('target_source') == 'derived_from_speaking_instances'
        for target in targets
    )
    return not derived


def resolve_question_conditions(request, profile, registry, *, actors=None, sounds=None,
                                moving=None, generator=None, include_compiled=False):
    """Compile this request's QA types, branches, instances and events.

    A request drives the sampler from its questions when it states
    ``qa_targets`` or ``question_branches``; a bare ``qa_ids`` list is still
    compiled and reported, but a mixed list of every type would otherwise fight
    over one Episode's knobs, so its knobs are recorded rather than applied.
    """
    if not isinstance(request,Mapping):
        return None
    targets=request.get('qa_targets')
    branches=request.get('question_branches')
    qa_ids=list(request.get('qa_ids') or [])
    if not targets and not qa_ids:
        return None
    from avengine.rooms.qa_episode import compile_question_conditions
    import sys
    rows=instance_rows(profile,request,actors,sounds,moving)
    result=compile_question_conditions(
        qa_ids,rows,targets=targets,branches=branches,registry=registry,
        task_family=request.get('task_family'),backend=request.get('backend'),
        generator=generator if generator is not None else sys.modules[__name__],
        public_time_precision=int(request.get('public_time_precision',0)),
        include_compiled=include_compiled,
        **({'question_mode': ((request.get('qa_sampling') or {}).get('acceptance_policy') or {}).get('question_mode')}
           if ((request.get('qa_sampling') or {}).get('acceptance_policy') or {}).get('question_mode') else {}))
    drives=_question_targets_drive_sampler(request)
    result['drives_sampler']=drives
    result['knob_application']=('applied_to_condition_profile' if drives else
                                'recorded_only_because_the_request_states_no_qa_target_or_branch')
    result['instances']=rows
    return result


# A hidden emitter can be hidden two ways, and the condition says so itself.
_HIDDEN_STATES = ('out_of_view', 'fully_occluded')


def _prefer_stated_visibility_route(knobs, request, questions):
    """Drop a compiled ``anchor_visibility`` when the caller states the other legal route.

    QA-20's ``none_of_them`` branch is satisfied by an emitter that is out of
    view **or** one that is fully occluded - its own evidence lists both states.
    The compiled knob named only ``off_screen``, so a request that asked for the
    occlusion route got ``anchor_visibility=off_screen`` bolted on top of
    ``pixel_occlusion_transition=fully_occluded`` and had to satisfy both at
    once, which nothing can. The alternative the condition already allows is
    honoured here instead of being overridden, and the question is not widened:
    only a value the evidence lists is accepted.
    """
    stated = request.get('profile') if isinstance(request.get('profile'), Mapping) else {}
    occlusion = stated.get('pixel_occlusion_transition')
    if not occlusion or 'anchor_visibility' not in knobs:
        return knobs
    if not str(occlusion).startswith('fully_occluded'):
        return knobs
    allows_occluded = False
    for entry in (questions or {}).get('compiled') or ():
        for item in (entry or {}).get('conditions') or ():
            if 'anchor_visibility' not in (item.get('planning') or {}):
                continue
            states = (item.get('evidence') or {}).get('visibility_state_in') or ()
            if 'fully_occluded' in states and 'out_of_view' in states:
                allows_occluded = True
    if not allows_occluded:
        return knobs
    knobs.pop('anchor_visibility', None)
    knobs['__visibility_route__'] = {
        'chosen': 'fully_occluded_stated_by_the_request',
        'dropped_compiled_knob': 'anchor_visibility',
        'both_allowed_by': 'the condition evidence lists out_of_view and fully_occluded',
    }
    return knobs


def resolve_conditioned_request(request, registry, *, condition_profile=None, generator=None):
    """Resolve the condition profile together with the compiled QA conditions.

    An explicitly supplied profile is never overridden, and an explicit
    ``request['profile']`` value always wins over a compiled knob, so a question
    can drive the sampler without silently replacing what the caller stated.
    """
    if condition_profile is not None:
        # The caller's profile wins, as it always has. What it may not do is
        # quietly disagree with the question: the compiled knobs were dropped
        # here and knob_application still read applied_to_condition_profile, so
        # an Episode solved from a base profile looked like one solved from the
        # branch. That is exactly how the CLI bypassed every compiled knob.
        profile=deepcopy(condition_profile)
        questions=resolve_question_conditions(request,profile,registry,generator=generator)
        if questions is not None:
            compiled=questions.get('sampler_profile') or {}
            owner={}
            for qa_id,by_qa in sorted((questions.get('sampler_profile_by_qa') or {}).items()):
                for knob in by_qa:
                    owner.setdefault(knob,qa_id)
            agreed,contradicted=[],[]
            for knob,value in sorted(compiled.items()):
                if knob not in profile:
                    contradicted.append({'kind':'profile_missing_compiled_knob','knob':knob,
                                         'requested':'absent from the supplied condition_profile',
                                         'compiled':value,
                                         'wanted_by':owner.get(knob,'a compiled question')})
                elif _profile_admits(profile[knob],value):
                    agreed.append(knob)
                else:
                    contradicted.append({'knob':knob,'requested':profile[knob],'compiled':value,
                                         'wanted_by':owner.get(knob,'a compiled question')})
            if questions.get('drives_sampler') and contradicted:
                raise ConditionedRequestConflict(contradicted)
            questions['knob_application']=(
                'caller_supplied_condition_profile_verified_against_compiled_knobs'
                if questions.get('drives_sampler') else
                'caller_supplied_condition_profile_and_the_request_states_no_qa_target_or_branch')
            questions['caller_profile_agrees_with_knobs']=agreed
            questions['caller_profile_unverified_knobs']=[row['knob'] for row in contradicted]
        return profile,questions
    base=resolve_condition_profile(request,registry)
    questions=resolve_question_conditions(request,base,registry,generator=generator)
    if (questions or {}).get('drives_sampler') and questions.get('conflicts'):
        # Two stated questions demanded different values for one knob. The
        # compiler already drops such a knob from ``sampler_profile``, which
        # would plan an Episode that answers neither of them.
        raise ConditionedRequestConflict([
            {'kind': 'two_questions', 'knob': row.get('knob'),
             'requested': row.get('values'), 'compiled': None,
             'wanted_by': sorted({
                 str(label).split('/')[0]
                 for labels in (row.get('values') or {}).values() for label in labels
             }) or 'two compiled questions'}
            for row in questions['conflicts']
        ])
    knobs=(questions or {}).get('sampler_profile') if (questions or {}).get('drives_sampler') else None
    if knobs:
        knobs=_prefer_stated_visibility_route(dict(knobs),request,questions)
    if knobs:
        # Name the QA type behind each knob so a refusal says which question
        # the stated profile is fighting, not just which knob.
        owner={}
        for qa_id,by_qa in sorted(((questions or {}).get('sampler_profile_by_qa') or {}).items()):
            for knob in by_qa:
                owner.setdefault(knob,qa_id)
        knobs=dict(knobs)
        route=knobs.pop('__visibility_route__',None)
        knobs['__wanted_by__']=owner
        if route is not None and questions is not None:
            questions['visibility_route_choice']=route
    profile=resolve_condition_profile(request,registry,question_knobs=knobs) if knobs else base
    if questions is not None and questions.get('drives_sampler'):
        profile['require_public_query_window']=bool(questions.get('requires_public_query_window'))
        profile['public_time_precision']=int(questions.get('public_time_precision',0))
    return profile,questions


def request_passthrough(request, clock, camera):
    """Record which clock, camera and speed values the request actually stated.

    A conditioned route may not quietly replace an explicit request, so the
    stated value and the applied value are recorded side by side.
    """
    stated=request.get('camera') if isinstance(request.get('camera'),Mapping) else {}
    motion=request.get('motion') if isinstance(request.get('motion'),Mapping) else {}
    rows={'camera':{},'clock':{},'motion':{}}
    # The request and the solved camera name the same quantity differently; the
    # pair is recorded so a route default can never pass for a stated value.
    for key,applied_key,alias in (('fov_deg','horizontal_fov_deg','camera_fov_deg'),
                                  ('height_above_floor_m','height_above_floor_m',None),
                                  ('resolution_hw','resolution_hw',None),
                                  ('motion','motion',None)):
        if key in stated or (alias and alias in request):
            rows['camera'][key]={'requested':deepcopy(stated.get(key,request.get(alias))),
                                 'applied':deepcopy(camera.get(applied_key)),'source':'request'}
        elif applied_key in camera:
            rows['camera'][key]={'requested':None,'applied':deepcopy(camera.get(applied_key)),
                                 'source':'sampler_default'}
    for key in ('frame_count','frame_rate_hz','sample_rate_hz'):
        rows['clock'][key]={'requested':deepcopy(request.get(key)),'applied':deepcopy(clock.get(key)),
                            'source':'request' if key in request else 'clock_default'}
    for key in ('speed_range_mps','walk_speed_range_mps'):
        if key in motion:
            rows['motion'][key]={'requested':deepcopy(motion[key]),'source':'request'}
    return rows


def _sampling_candidate_index(request):
    value = (
        request.get("sampling_candidate_index", 0)
        if isinstance(request, Mapping)
        else 0
    )
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            "sampling_candidate_index must be a nonnegative integer"
        )
    value = int(value)
    if value < 0:
        raise ValueError(
            "sampling_candidate_index must be a nonnegative integer"
        )
    return value


def _sampling_provenance(seed, attempt, sampling_candidate_index):
    """Return the exact RNG provenance for one planner attempt.

    Candidate index zero, including an omitted field, deliberately retains the
    historical seed/attempt stream. Keeping construction and consumption in
    one helper prevents plan serialization from referring to a loop-local seed
    variable that is absent on an alternate StageContext path.
    """
    components = [int(seed), int(attempt), 20260906]
    if sampling_candidate_index:
        components.append(int(sampling_candidate_index))
    return {
        'sampling_candidate_index': int(sampling_candidate_index),
        'index_applied': bool(sampling_candidate_index),
        'seed_components': components,
        'policy': (
            'legacy_seed_attempt_stream_v1'
            if not sampling_candidate_index
            else 'seed_attempt_candidate_index_stream_v1'
        ),
    }


def build_conditioned_plan(*, room, request, source_registry, sounds, space, mesh,
                           clock, condition_profile=None, region=None, generator=None):
    sampling_candidate_index = _sampling_candidate_index(request)
    profile,questions=resolve_conditioned_request(request,source_registry,
                                                  condition_profile=condition_profile,
                                                  generator=generator)
    seed=int(request.get('seed',0));failures=Counter();last_errors={};first_errors={};stage_reasons={}
    static_placement_plan = request.get('_static_source_placement_plan')
    static_placement_rows = _static_placement_rows(static_placement_plan)
    for attempt in range(profile['retry_budget_within_profile']):
        # Index 0 keeps the historical SeedSequence exactly. Nonzero indices
        # select a sibling candidate without changing the resolved profile.
        sampling_provenance = _sampling_provenance(
            seed, attempt, sampling_candidate_index
        )
        rng=np.random.default_rng(
            np.random.SeedSequence(sampling_provenance['seed_components'])
        )
        try:
            actors=select_entities(request,profile,source_registry,rng)
            actor_ids = {str(actor['entity_instance_id']) for actor in actors}
            unknown_placements = sorted(
                set(static_placement_rows) - actor_ids
            )
            if unknown_placements:
                raise CandidateFailure(
                    'placement', 'static_source_placement_instance_not_selected'
                )
            for actor in actors:
                placement = static_placement_rows.get(
                    str(actor['entity_instance_id'])
                )
                if placement is None:
                    continue
                if actor['entity_class'] not in RIGID:
                    raise CandidateFailure(
                        'placement', 'static_source_placement_requires_static_asset'
                    )
                if str(placement.get('asset_id')) != str(actor.get('asset_id')):
                    raise CandidateFailure(
                        'placement', 'static_source_placement_asset_mismatch'
                    )
                actor['static_placement'] = deepcopy(placement)
            selected_sounds=select_sounds(actors,sounds,profile,clock,request,rng)
            question_check=None
            motion_context=None
            visibility_requirements=()
            scheduled_events=None
            fixed_schedule=None
            if questions is not None:
                question_check=resolve_question_conditions(
                    request,profile,source_registry,actors=actors,sounds=selected_sounds,
                    generator=generator,include_compiled=True)
                compiled_conditions=question_check.pop('_compiled_conditions', ())
                question_drives_sampler = bool(
                    question_check.get('drives_sampler')
                )
                if question_drives_sampler:
                    visibility_requirements = _visibility_requirements_for_selection(
                        compiled_conditions, actors, profile, int(clock['frame_count'])
                    )
                if (question_drives_sampler
                        and questions.get('status')=='candidate'
                        and question_check.get('status')=='unsupported'):
                    raise CandidateFailure('question','compiled_conditions_unavailable_for_selected_instances')
                if (
                    question_drives_sampler
                    and question_check.get('status') == 'candidate'
                    and compiled_conditions
                ):
                    scheduled_events,fixed_schedule,event_start_s,other_event_windows_s=(
                        _schedule_for_motion_solver(selected_sounds,profile,clock,rng))
                    motion_context=_solve_motion_conditions(
                        request,profile,source_registry,actors,selected_sounds,
                        compiled_conditions,clock,event_start_s,other_event_windows_s)
                    if motion_context is not None:
                        profile.setdefault(
                            'end_hold_s', motion_context['budget'].end_hold_s)
            try:
                paths,rotations,moving,emitters,bodies,route_record=sample_routes(space,actors,profile,clock,rng,region,
                    required_windows={i:[selected_sounds[i]['audible_start_sample'],selected_sounds[i]['audible_end_sample_exclusive']] for i in profile['anchor_indices']},
                    room=room,
                    motion_requirements=(motion_context or {}).get('requirements'),
                    motion_budget=(motion_context or {}).get('budget'),
                    constructive_motion=_constructive_motion_requested(
                        request, profile
                    ),
                    static_placements=static_placement_plan,
                    visibility_requirements=visibility_requirements)
            except CandidateFailure:
                raise
            except ValueError as exc:
                raise CandidateFailure('routes','navigation_sampling_failed: '+str(exc)) from exc
            camera,events,conditions=select_camera_and_schedule(space,mesh,paths,moving,emitters,bodies,actors,
                                                                selected_sounds,profile,clock,request,rng,region,
                                                                event_bindings=scheduled_events,
                                                                fixed_schedule=fixed_schedule,
                                                                motion_requirements=(motion_context or {}).get('requirements'),
                                                                motion_solution=(motion_context or {}).get('primary'),
                                                                visibility_requirements=visibility_requirements,
                                                                room_package=(
                                                                    room.get('room_package')
                                                                    if isinstance(room, Mapping)
                                                                    else None
                                                                ),
                                                                static_placements=static_placement_plan,
                                                                root_rotations=rotations,
                                                                source_registry=source_registry)
            if motion_context is not None:
                conditions['motion_solver']=_motion_context_record(motion_context)
        except CandidateFailure as exc:
            # Every distinct reason is kept: a later inapplicable candidate must
            # not overwrite the real error an earlier attempt hit.
            failures[exc.stage+':'+exc.reason]+=1;last_errors[exc.stage]=exc.reason
            first_errors.setdefault(exc.stage,exc.reason)
            reasons=stage_reasons.setdefault(exc.stage,{})
            reasons[exc.reason]=reasons.get(exc.reason,0)+1
            continue
        frames=[]; action_ticks=[int(rng.integers(int(a['timeline']['walk_phase_period_frames'])))*int(clock['ticks_per_frame']) if a.get('timeline') else 0 for a in actors]
        for f in range(int(clock['frame_count'])):
            states=[]
            for i,actor in enumerate(actors):
                timeline=actor.get('timeline');motion=bool(moving[i,f])
                action=(timeline['walking_action_id'] if motion else timeline['idle_action_id']) if timeline else 'static'
                if timeline and motion:action_ticks[i]+=int(clock['ticks_per_frame'])
                tick=action_ticks[i] if timeline and motion else 0
                period=int(timeline['walk_phase_period_frames'])*int(clock['ticks_per_frame']) if timeline else 1
                phase=(tick%period)/period if timeline and motion else 0.
                placement = static_placement_rows.get(
                    str(actor['entity_instance_id'])
                )
                if placement is not None and not _placement_is_ground(placement):
                    root_matrix = deepcopy(
                        placement['root_transform']['matrix_row_major']
                    )
                    emitter_transform = deepcopy(
                        placement['emitter_transform']
                    )
                    support_identity = deepcopy(
                        placement.get('support_identity') or {}
                    )
                else:
                    root_matrix = _root_matrix_from_pose(
                        paths[i,f], rotations[i,f]
                    )
                    emitter_transform = {
                        'position_m': emitters[i,f].tolist(),
                    }
                    support_identity = None
                state={'actor_id':actor['actor_id'],'entity_instance_id':actor['entity_instance_id'],
                       'root_transform':{'matrix_row_major':root_matrix,
                                         'translation_m':paths[i,f].tolist(),
                                         'rotation_xyzw':rotations[i,f].tolist(),
                                         'scale':[1.,1.,1.]},
                       'action_id':action,'action_phase':phase,'action_time_ticks':tick,'moving':motion,
                       'planned_emitter_m':emitters[i,f].tolist(),
                       'emitter_transform':emitter_transform,
                       'support_identity':support_identity,
                       'frame_index':f}
                states.append(state)
            frames.append({'frame_index':f,'pts_ticks':f*clock['ticks_per_frame'],'actor_states':states,'camera_state':deepcopy(camera)})
        bindings=[deepcopy(selected_sounds[i]) for i in sorted(selected_sounds)]
        instances=instance_rows(profile,request,actors,selected_sounds,moving)
        if question_check is not None:
            question_check['instances']=deepcopy(instances)
        request_for_plan = deepcopy(dict(request))
        request_for_plan.pop('_static_source_placement_plan', None)
        plan={'kind':'avengine_question_driven_episode','plan_coordinates':'renderer_neutral','coordinate_frame':dict(COORDINATE_FRAME),
              'episode_id':str(request['episode_id']),'seed':seed,'status':'research_candidate','clock':deepcopy(dict(clock)),
              'scene':{'scene_id':room.get('scene_id',room['room_id']),'room_id':room['room_id']},'request':request_for_plan,
              'static_source_placements':(
                  deepcopy(static_placement_plan)
                  if static_placement_plan is not None
                  else None
              ),
              'sampling_provenance':deepcopy(sampling_provenance),
              'condition_profile':profile,'planned_conditions':conditions,'activity_plan':route_record,
              'camera_condition_sampling':conditions,'audio_events':events,'voice_bindings':bindings,
              'entity_instances':instances,
              'instance_role_map':{row['entity_instance_id']:{'actor_id':row['actor_id'],
                    'source_endpoint_id':row['source_endpoint_id'],'role':row['role'],
                    'is_anchor':row['is_anchor'],'speaking':row['speaking']} for row in instances},
              'request_passthrough':request_passthrough(request,clock,camera),
              'visual_plan':{'backend_role':'production_visual','camera':camera,'actors':actors,'frames':frames,
                             'render':{'frame_count':clock['frame_count'],'fps_num':clock['frame_rate_hz'],'fps_den':1,'ticks_per_frame':clock['ticks_per_frame']},
                             'authority':{'actor_state':'avengine_conditioned_static_sampler','camera_listener':'avengine_conditioned_static_sampler','backend_may_replan':False}},
              'resources':deepcopy(dict(room)),
              'question_condition_match':deepcopy(question_check) if question_check is not None else {
                  'candidate_qa_ids':list(request.get('qa_ids',list(QA_IDS))),'status':'candidate',
                  'episode_validity':'not_run','compiled':[],
                  'claim_boundary':'no qa_target or qa_ids stated, so no condition was compiled'},
              'room_capabilities':{'status':'potential_only','evidence_refs':{'navigation':deepcopy(space.metadata)}},
              'planning_result':{'status':'research_candidate','attempts':attempt+1,'condition_profile':profile,'failure_histogram':dict(failures),
                                  'sampling_provenance':deepcopy(sampling_provenance),
                                  'motion_solver':_motion_context_record(motion_context)},
              'evidence_status':{'native_visual':'not_run','native_audio':'not_run','qa_validity':'not_run'},'qualification_claim':False,
              'formal_dataset_registration_authorized':False}
        return plan
    raise ConditionedPlanningFailure({'status':'failed','condition_profile':profile,'attempts':profile['retry_budget_within_profile'],
                                      'failure_histogram':dict(failures),'last_errors_by_stage':last_errors,
                                      'first_errors_by_stage':first_errors,
                                      'errors_by_stage':{stage:dict(reasons) for stage,reasons in stage_reasons.items()},
                                      'question_conditions':deepcopy(questions) if questions is not None else None,
                                      'gap_category':'evidence_missing_or_unsampled'})


def _prepare_static_placement_plan(request, source_registry, room):
    """Resolve qualification-referenced support evidence before sampling.

    The support catalog is an observed visual resource. Its measured
    asset-local geometry is copied by asset ID into each placement request;
    caller-supplied geometry cannot replace that catalog evidence.
    """
    spec = (
        request.get("static_source_placement")
        or request.get("static_source_placements")
        or request.get("placement_execution")
    )
    placement_report = request.get("placement_plan")
    if spec is None and isinstance(placement_report, Mapping):
        spec = (
            placement_report.get("static_source_placement")
            or placement_report.get("placement_execution")
            or placement_report.get("execution")
        )
        if spec is None:
            report_instances = placement_report.get("instances") or ()
            non_ground_report = any(
                isinstance(row, Mapping)
                and (
                    str(
                        (row.get("placement") or {}).get(
                            "requested_surface", ""
                        )
                    ).lower()
                    in {"tabletop", "wall", "ceiling"}
                    or (
                        isinstance(
                            (row.get("placement") or {}).get(
                                "support_planning"
                            ),
                            Mapping,
                        )
                        and (row.get("placement") or {}).get(
                            "support_planning", {}
                        ).get("support_surface_id")
                    )
                )
                for row in report_instances
            )
            if non_ground_report:
                raise ValueError(
                    "placement_plan declares non-ground static sources but "
                    "has no static placement execution block"
                )
    if spec is None:
        return None
    if not isinstance(spec, Mapping):
        raise ValueError("static_source_placement must be a mapping")
    catalog_path = (
        spec.get("catalog_path")
        or spec.get("support_surface_catalog")
        or spec.get("catalog_ref")
    )
    if not isinstance(catalog_path, str) or not catalog_path:
        raise ValueError(
            "static_source_placement.catalog_path is required"
        )
    from pathlib import Path
    from avengine.rooms.qa_episode import read_json
    from avengine.rooms.source_placement import (
        SOURCE_PLACEMENT_SCHEMA,
        plan_static_source_placements,
    )

    catalog = read_json(catalog_path)
    if catalog.get("schema") != "avengine_support_surface_catalog_v1":
        raise ValueError("static placement catalog schema is unsupported")
    catalog_room = catalog.get("room") or {}
    catalog_room_id = catalog_room.get("room_id")
    requested_room_id = (
        room.get("room_id")
        if isinstance(room, Mapping)
        else None
    ) or request.get("room_id")
    if (
        requested_room_id is not None
        and catalog_room_id is not None
        and str(requested_room_id) != str(catalog_room_id)
    ):
        raise ValueError(
            "static placement catalog room does not match request room"
        )
    measurements = catalog.get("asset_visual_geometry_measurements")
    if not isinstance(measurements, Mapping):
        raise ValueError(
            "static placement catalog has no asset_visual_geometry_measurements"
        )

    qualification_path = spec.get("qualification_config")
    qualification_episode_id = spec.get("qualification_episode_id")
    qualification_instances = None
    if qualification_path is not None:
        if not isinstance(qualification_path, str) or not qualification_path:
            raise ValueError(
                "static placement qualification_config must be a path"
            )
        qualification = read_json(qualification_path)
        episodes = qualification.get("episodes")
        if not isinstance(episodes, Sequence):
            raise ValueError(
                "qualification config has no episodes list"
            )
        if qualification_episode_id is None:
            qualification_episode_id = request.get(
                "qualification_episode_id"
            )
        if qualification_episode_id is None:
            raise ValueError(
                "qualification_episode_id is required when deriving placement requests"
            )
        matching = [
            row for row in episodes
            if isinstance(row, Mapping)
            and str(row.get("episode_id")) == str(qualification_episode_id)
        ]
        if len(matching) != 1:
            raise ValueError(
                "qualification_episode_id must resolve exactly one episode"
            )
        placement_plan = (
            matching[0].get("request_extras", {})
            if isinstance(matching[0].get("request_extras"), Mapping)
            else {}
        ).get("placement_plan")
        qualification_instances = (
            placement_plan.get("instances")
            if isinstance(placement_plan, Mapping)
            else None
        )
        if not isinstance(qualification_instances, Sequence):
            raise ValueError(
                "qualification episode has no placement_plan.instances"
            )

    raw_requests = spec.get("requests")
    if raw_requests is None:
        raw_requests = qualification_instances
    if not isinstance(raw_requests, Sequence) or isinstance(
        raw_requests, (str, bytes)
    ):
        raise ValueError(
            "static_source_placement.requests must be a list or derive from qualification config"
        )
    selected_assets = {
        str(value)
        for value in (request.get("source_asset_ids") or ())
        if value is not None
    }
    requests = []
    registry_assets = {
        str(row.get("asset_id")): row
        for row in (source_registry.get("assets") or ())
        if isinstance(row, Mapping) and row.get("asset_id") is not None
    }
    for raw in raw_requests:
        if not isinstance(raw, Mapping):
            raise ValueError(
                "static placement request rows must be mappings"
            )
        nested = raw.get("placement")
        nested = nested if isinstance(nested, Mapping) else {}
        asset_id = raw.get("asset_id")
        if asset_id is None:
            asset_id = nested.get("asset_id")
        if asset_id is None:
            continue
        asset_id = str(asset_id)
        if selected_assets and asset_id not in selected_assets:
            continue
        support_id = raw.get("support_surface_id")
        if support_id is None:
            support_id = nested.get("support_surface_id")
        support_planning = nested.get("support_planning")
        if support_id is None and isinstance(support_planning, Mapping):
            support_id = support_planning.get("support_surface_id")
        item = {
            "instance_id": raw.get("instance_id"),
            "asset_id": asset_id,
            "support_surface_id": support_id,
            "yaw_deg": raw.get(
                "yaw_deg", nested.get("yaw_deg", 0.0)
            ),
        }
        if "candidate_index" in raw or "candidate_index" in nested:
            item["candidate_index"] = raw.get(
                "candidate_index", nested.get("candidate_index")
            )
        record = registry_assets.get(asset_id)
        if record is not None and raw.get("asset_revision") is not None:
            item["asset_revision"] = raw["asset_revision"]
        measured = measurements.get(asset_id)
        if measured is None:
            raise ValueError(
                "static placement asset has no catalog visual geometry measurement: "
                + asset_id
            )
        item["asset_geometry"] = deepcopy(dict(measured))
        requests.append(item)
    if not requests:
        raise ValueError(
            "static placement produced no requests for selected assets"
        )
    config = spec.get("config") or spec.get("placement_config")
    if not isinstance(config, Mapping):
        raise ValueError(
            "static_source_placement.config is required"
        )
    visual_geometry = catalog.get("visual_geometry")
    layout = catalog.get("layout")
    if not isinstance(layout, Mapping) or not isinstance(
        visual_geometry, Mapping
    ):
        raise ValueError(
            "static placement catalog layout/visual_geometry is incomplete"
        )
    room_data = {
        "room_id": catalog_room_id
        or requested_room_id,
        "scene_glb": catalog_room.get("scene_glb"),
        "coordinate_frame": catalog_room.get("coordinate_frame"),
    }
    result = plan_static_source_placements(
        source_registry,
        room_data,
        layout,
        visual_geometry,
        requests,
        config=config,
    )
    if result.get("schema") != SOURCE_PLACEMENT_SCHEMA:
        raise ValueError("static placement helper returned an unexpected schema")
    if result.get("status") != "planned" or any(
        row.get("status") != "planned"
        for row in result.get("instances", ())
        if isinstance(row, Mapping)
    ):
        raise ValueError(
            "static placement rejected support request: "
            + str(result)
        )
    return {
        **result,
        "catalog_ref": str(Path(catalog_path).expanduser().resolve()),
        "catalog_schema": catalog.get("schema"),
        "qualification_config_ref": (
            None
            if qualification_path is None
            else str(Path(qualification_path).expanduser().resolve())
        ),
        "qualification_episode_id": qualification_episode_id,
        "config": deepcopy(dict(config)),
    }


def solve_conditioned_episode(*, request, source_registry, sounds, room=None, room_id=None,
                             space=None, mesh=None, clock=None, condition_profile=None,
                             region=None, generator=None):
    """Solve one Episode from a config-shaped request: the entry a stage calls.

    It resolves the room's planning resources when they are not supplied,
    resolves the condition profile together with the compiled question
    conditions, solves them, and returns the plan with its instance/role map.
    It runs no native stage and asserts no evidence: every ``evidence_status``
    the plan carries stays ``not_run``.
    """
    from avengine.rooms.furniture_layout import clock_config
    layout = None
    effective = deepcopy(dict(request))
    if space is None or mesh is None:
        from avengine.capture.qa_plan_adapters import (
            load_planning_resources, load_planning_resources_for_room,
        )
        selected = room_id or effective.get('room_id')
        if selected and effective.get('room_catalog'):
            space, mesh, layout, resolution = load_planning_resources_for_room(str(selected), effective)
            room = room or dict(resolution.planning_room)
        elif room is not None:
            space, mesh, layout = load_planning_resources(dict(room), effective)
        else:
            raise ValueError('solve_conditioned_episode needs a room, a room_id with a '
                             'room_catalog, or an already loaded space and mesh')
    placement_plan = _prepare_static_placement_plan(
        effective, source_registry, room
    )
    if placement_plan is not None:
        effective['_static_source_placement_plan'] = placement_plan
    if clock is None:
        clock = clock_config(frame_count=int(effective.get('frame_count', 240)),
                             frame_rate_hz=float(effective.get('frame_rate_hz', 15)),
                             sample_rate_hz=int(effective.get('sample_rate_hz', 16000)))
    camera = effective.setdefault('camera', {})
    if layout is not None:
        camera.setdefault('resolution_hw', layout.get('capture_resolution_hw', [720, 1280]))
    plan = build_conditioned_plan(
        room=room or {}, request=effective, source_registry=source_registry, sounds=sounds,
        space=space, mesh=mesh, clock=clock, condition_profile=condition_profile,
        region=region if region is not None else effective.get('planning_region_m'),
        generator=generator)
    if layout is not None:
        plan['visual_lighting'] = deepcopy(layout.get('visual_lighting', {}))
    return {'plan': plan, 'layout': layout, 'space': space,
            'static_source_placements': deepcopy(placement_plan),
            'condition_profile': plan['condition_profile'],
            'question_conditions': plan['question_condition_match'],
            'entity_instances': plan['entity_instances'],
            'instance_role_map': plan['instance_role_map'],
            'claim_boundary': 'a solved neutral plan; native visual, audio and answer '
                              'validity all remain not_run'}


def _gain_metadata(value: Any) -> dict[str, Any]:
    """Copy gain/normalization fields without interpreting policy labels."""
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        name = str(key)
        folded = name.casefold()
        if folded == "source_normalization":
            continue
        if any(token in folded for token in ("gain", "normal", "peak")) or folded in {
            "target_dbfs", "target_peak_dbfs"
        }:
            result[name] = deepcopy(item)
    return result


def source_normalization_metadata(
    raw: Mapping[str, Any],
    *,
    related: Sequence[Mapping[str, Any]] = (),
    container: Mapping[str, Any] | None = None,
    measured_peak_dbfs: float | None = None,
    measured_peak_source: str | None = None,
    source_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Carry source-level normalization facts separately from runtime gain."""
    records: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(raw, Mapping):
        records.append(("raw", raw))
        facts = raw.get("facts")
        if isinstance(facts, Mapping):
            records.append(("facts", facts))
    for index, value in enumerate(related):
        if isinstance(value, Mapping):
            records.append((f"related_{index}", value))
    if isinstance(container, Mapping):
        records.append(("container", container))

    existing = raw.get("source_normalization") if isinstance(raw, Mapping) else None
    result = deepcopy(existing) if isinstance(existing, Mapping) else {}

    policy = None
    policy_found = False
    target = None
    target_found = False
    applied_gain = None
    applied_gain_found = False
    normalized = None
    normalized_found = False
    measured = measured_peak_dbfs
    measured_source = measured_peak_source
    if measured is None:
        for label, record in records:
            for key in ("measured_peak_dbfs", "prepared_peak_dbfs", "peak_dbfs"):
                if key in record:
                    measured = deepcopy(record[key])
                    measured_source = measured_source or f"{label}.{key}"
                    break
            if measured is not None:
                break
    for _label, record in records:
        if not policy_found and "normalization_policy" in record:
            policy = deepcopy(record["normalization_policy"])
            policy_found = True
        if not target_found:
            candidate = record.get("normalization_policy")
            if isinstance(candidate, Mapping) and "target_dbfs" in candidate:
                target = deepcopy(candidate["target_dbfs"])
                target_found = True
            elif "target_dbfs" in record:
                target = deepcopy(record["target_dbfs"])
                target_found = True
            elif "target_peak_dbfs" in record:
                target = deepcopy(record["target_peak_dbfs"])
                target_found = True
        if not applied_gain_found and "applied_gain_db" in record:
            applied_gain = deepcopy(record["applied_gain_db"])
            applied_gain_found = True
        if not normalized_found and "normalization_applied" in record:
            normalized = deepcopy(record["normalization_applied"])
            normalized_found = True

    if policy_found:
        result["policy"] = policy
    elif "policy" not in result:
        result["policy"] = None
    if target_found:
        result["target_dbfs"] = target
    elif "target_dbfs" not in result:
        result["target_dbfs"] = None
    if applied_gain_found:
        result["applied_gain_db"] = applied_gain
    elif "applied_gain_db" not in result:
        result["applied_gain_db"] = None
    if normalized_found:
        result["normalization_applied"] = normalized
    elif "normalization_applied" not in result:
        result["normalization_applied"] = None
    if measured is not None or "measured_peak_dbfs" not in result:
        result["measured_peak_dbfs"] = measured
    if measured_source is not None or "measured_peak_source" not in result:
        result["measured_peak_source"] = measured_source

    source = deepcopy(result.get("source")) if isinstance(result.get("source"), Mapping) else {}
    for _label, record in records:
        for key in (
            "source", "source_pcm_path", "source_relative", "source_sha256",
            "source_metadata_path", "source_metadata_manifest", "source_origin",
            "source_origin_aliases", "source_event_registry",
        ):
            if key in record and key not in source:
                source[key] = deepcopy(record[key])
        dry_audio = record.get("dry_audio")
        if isinstance(dry_audio, Mapping):
            for key, target_key in (("uri", "dry_audio_uri"), ("sha256", "dry_audio_sha256")):
                if key in dry_audio and target_key not in source:
                    source[target_key] = deepcopy(dry_audio[key])
    if source_overrides:
        source.update(deepcopy(dict(source_overrides)))
    result["source"] = source

    metadata = deepcopy(result.get("metadata")) if isinstance(result.get("metadata"), Mapping) else {}
    for label, record in records:
        subset = _gain_metadata(record)
        if subset:
            metadata[label] = subset
    if metadata:
        result["metadata"] = metadata
    return result


def load_conditioned_sound_pool(payload, *, source_path=None):
    """Bridge P7 crop-relative activity once; preserve source gain facts."""
    from pathlib import Path
    import wave
    from types import SimpleNamespace
    prepared=isinstance(payload,Mapping) and 'prepared_set_id' in payload
    rows=payload.get('clips',[]) if prepared else payload.get('sounds',[]) if isinstance(payload,Mapping) else payload
    root=Path(source_path).resolve().parent if source_path else Path.cwd()
    result=[]
    source_manifest = {"manifest_path": str(Path(source_path).resolve())} if source_path else None
    for raw in rows:
        if prepared and raw.get('status')!='prepared':continue
        sound=deepcopy(raw)
        path=Path(raw['prepared'] if prepared else raw['path']).expanduser()
        if not path.is_absolute():path=root/path
        with wave.open(str(path), 'rb') as wav:
            info=SimpleNamespace(frames=wav.getnframes(), samplerate=wav.getframerate(), channels=wav.getnchannels())
        if info.channels!=1:raise ValueError('conditioned sound pool must be real mono source PCM')
        sound.update(path=str(path.resolve()),sample_count=int(info.frames),sample_rate_hz=int(info.samplerate))
        if isinstance(raw.get('source_normalization'), Mapping):
            # This namespace describes prior source processing. Carrier-level
            # runtime gain or normalization flags must not overwrite it.
            sound['source_normalization'] = deepcopy(raw['source_normalization'])
            if source_path:
                source = sound['source_normalization'].setdefault('source', {})
                source['input_manifest_path'] = str(Path(source_path).resolve())
        else:
            sound['source_normalization'] = source_normalization_metadata(
                raw, container=payload if isinstance(payload, Mapping) else None,
                source_overrides=source_manifest,
            )
        if prepared:
            facts=raw['facts'];source_rate=int(facts['source_rate_hz']);offset=int(raw['source_crop_start_sample'])
            intervals=[[int(round((i['start_sample']-offset)*info.samplerate/source_rate)),
                        int(round((i['end_sample_exclusive']-offset)*info.samplerate/source_rate))] for i in raw['source_activity']['intervals']]
            sound.update(sound_asset_id=raw['prepared_audio_id'],sound_class='speech',gender=raw['gender'],
                         transcript=raw.get('transcript'),source_activity_intervals_samples=intervals,
                         audible_start_sample=min(i[0] for i in intervals),audible_end_sample_exclusive=max(i[1] for i in intervals),
                         active_duration_s=float(raw['source_activity']['active_duration_s']),
                         activity_coordinate='prepared_clip_samples',prepared_review=raw.get('human_review'),
                         transcript_scope='source_transcript_for_cropped_excerpt')
            # Keep traceability without copying bulky detector/source-stage records per event.
            for key in ('facts','source_activity','activity_filter_parameters','detector_parameters','filter_parameters'):
                sound.pop(key,None)
        else:
            sound.setdefault('sound_class','speech' if sound.get('transcript') else None)
            intervals=sound.get('source_activity_intervals_samples')
            if not intervals:
                raise ValueError('conditioned sounds require measured source_activity_intervals_samples')
            sound.setdefault('audible_start_sample',min(i[0] for i in intervals))
            sound.setdefault('audible_end_sample_exclusive',max(i[1] for i in intervals))
            sound.setdefault('active_duration_s',sum(b-a for a,b in intervals)/info.samplerate)
        if not 0<=sound['audible_start_sample']<sound['audible_end_sample_exclusive']<=info.frames:
            raise ValueError('activity/crop offsets do not resolve inside prepared PCM')
        result.append(sound)
    if not result:raise ValueError('conditioned sound pool has no prepared candidates')
    return result
