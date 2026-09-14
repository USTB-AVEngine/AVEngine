"""Replay the registered-appearance review over captures that already exist.

Changing how registered appearance is read from pixels is only defensible if
the change can be measured on the captures the chain has already produced. This
module re-runs the review against the retained frames and instance masks of any
number of rendered episodes and reports what changed, grouped by room, without
re-rendering anything and without writing into the episode trees it reads.

Two measurements come out of one pass. The first is the confusion between the
registered value and the verdict, which says how much evidence the chain gets.
The second is cross acceptance: every actor is also offered the other values of
its own kind, and any value that is accepted while not registered is a false
acceptance. A classifier that simply says yes more often shows up immediately in
the second table, which is why both are always reported together.
"""
from __future__ import annotations

from collections import Counter
import inspect as _inspect
import json
import re
from pathlib import Path
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from avengine.rooms.qa_evidence import _mask_frame_row, _modal_array, _rgb_frame
from avengine.rooms.qa_evidence import (
    bbox_touches_frame_edge,
    inspect_registered_appearance,
    nonhuman_appearance_placeholder_thresholds,
)

REPLAY_SCHEMA = "avengine_qa_appearance_replay_v1"

VISIBLE_STATES = frozenset({"visible_clear", "visible_occluded"})

#: The values a cross-acceptance probe offers an actor of each kind. They are
#: the registered vocabulary of that kind, so the probe asks exactly the question
#: a question generator would: could this actor have been certified as something
#: it is not?
CROSS_ACCEPTANCE_VALUES: dict[str, tuple[str, ...]] = {
    "human": ("blue", "green", "yellow", "burgundy", "pink", "white"),
    "device": ("white", "black", "light_gray", "silver", "beige", "walnut_veneer"),
    "animal": (
        "standard_yellow",
        "standard_blue",
        "standard_red",
        "standard_sable",
        "standard_black_white",
        "standard_white_tan",
    ),
}

#: The four controlled shirt assets whose registered value is authoritative by
#: construction; a cross acceptance among these four is a hard failure.
CONTROLLED_TOP_COLORS = ("green", "yellow", "blue", "burgundy")


def _inspect_kwargs(thresholds: Mapping[str, Any]) -> dict[str, Any]:
    """Only pass the knobs the installed classifier actually declares.

    The point of a replay is to compare one revision of the classifier against
    another on the same pixels, so the tool has to run against a revision that
    predates any knob added with the change under measurement.
    """
    offered = {
        "minimum_color_pixels": int(thresholds["minimum_color_pixels"]),
        "dominance_ratio": float(thresholds["dominance_ratio"]),
        "color_component_fractions": thresholds.get("color_component_fractions"),
        "value_minimum_shares": thresholds.get("value_minimum_shares"),
    }
    accepted = _inspect.signature(inspect_registered_appearance).parameters
    return {name: value for name, value in offered.items() if name in accepted}


_SEED_IN_NAME = re.compile(r"s(\d+)\b")


def episode_seed(episode_name: str) -> int | None:
    """The sampling seed an episode directory name carries, when it carries one."""
    found = _SEED_IN_NAME.findall(str(episode_name))
    return int(found[-1]) if found else None


def seed_group(episode_name: str) -> str:
    """Split episodes into a tuning half and a held-out half by seed parity.

    Nothing about a seed's parity correlates with what a room looks like, so the
    split is arbitrary with respect to the measurement and can be declared before
    the numbers are read. Thresholds are read off the tuning half; the held-out
    half is what the false-acceptance claim rests on.
    """
    seed = episode_seed(episode_name)
    if seed is None:
        return "unseeded"
    return "tuning" if seed % 2 == 0 else "holdout"


def episode_directories(root: Path) -> list[Path]:
    """Every episode directory under a render root, found by its own products."""
    found = {
        path.parent.parent
        for path in Path(root).rglob("capture/pixel_visibility_truth.json")
    }
    return sorted(found)


def _appearance_review_path(episode: Path) -> Path | None:
    refs = episode / "delivery" / "input_refs.json"
    if refs.is_file():
        try:
            value = json.loads(refs.read_text(encoding="utf-8")).get("appearance_review")
        except (OSError, ValueError):
            value = None
        if isinstance(value, str) and Path(value).is_file():
            return Path(value)
    local = episode / "delivery" / "appearance_review.json"
    return local if local.is_file() else None


def _room_id(episode: Path) -> str:
    plan = episode / "plan" / "episode_plan.json"
    if plan.is_file():
        try:
            scene = json.loads(plan.read_text(encoding="utf-8")).get("scene")
        except (OSError, ValueError):
            scene = None
        if isinstance(scene, Mapping):
            for key in ("room_id", "scene_id"):
                if isinstance(scene.get(key), str) and scene[key].strip():
                    return str(scene[key]).strip()
    name = episode.parent.name if episode.name == "episode" else episode.name
    return name.split("__")[0] or "unknown"


def _episode_name(episode: Path) -> str:
    return episode.parent.name if episode.name == "episode" else episode.name


def _selected_frames(frame_indices: Sequence[int], stride: int) -> list[int]:
    step = max(1, int(stride))
    chosen = list(frame_indices[::step])
    if frame_indices and frame_indices[-1] not in chosen:
        chosen.append(int(frame_indices[-1]))
    return sorted({int(value) for value in chosen})


def replay_episode(
    episode: Path,
    *,
    frame_stride: int = 10,
    cross_acceptance_frames: int = 2,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-run the appearance review for one already-rendered episode."""
    episode = Path(episode).resolve()
    capture = episode / "capture"
    truth = json.loads((capture / "pixel_visibility_truth.json").read_text(encoding="utf-8"))
    review_path = _appearance_review_path(episode)
    stored = json.loads(review_path.read_text(encoding="utf-8")) if review_path else {}
    stored_actors = stored.get("actors") if isinstance(stored.get("actors"), Mapping) else {}
    resolved = dict(thresholds) if thresholds else nonhuman_appearance_placeholder_thresholds()
    knobs = _inspect_kwargs(resolved)
    frame_indices = truth.get("frame_indices")
    if not isinstance(frame_indices, list) or not frame_indices:
        raise ValueError(f"pixel visibility truth has no frame indices: {episode}")
    resolution = truth.get("resolution_hw")
    selected = _selected_frames([int(value) for value in frame_indices], frame_stride)
    rgb_path = capture / "rgb.npy"
    rgb_array = np.load(rgb_path, mmap_mode="r", allow_pickle=False) if rgb_path.is_file() else None
    masks = capture / "native_pixel_masks_depth_authority_v1.npz"
    room = _room_id(episode)
    actors: dict[str, Any] = {}
    started = time.monotonic()
    with np.load(masks, allow_pickle=False) as data:
        modal = _modal_array(data)
        if not isinstance(resolution, Sequence) or isinstance(resolution, (str, bytes)) or len(resolution) != 2:
            resolution = [int(modal.shape[1]), int(modal.shape[2])]
        resolution_hw = [int(resolution[0]), int(resolution[1])]
        work: dict[int, list[tuple[str, Mapping[str, Any]]]] = {}
        for actor_id, instance in (truth.get("per_instance") or {}).items():
            if not isinstance(instance, Mapping):
                continue
            record = stored_actors.get(str(actor_id))
            record = record if isinstance(record, Mapping) else {}
            value = record.get("value") or record.get("attribute_value")
            frames = instance.get("frames") if isinstance(instance.get("frames"), list) else []
            visible = [
                frame for frame in frames
                if isinstance(frame, Mapping) and frame.get("state") in VISIBLE_STATES
                and int(frame.get("visible_pixels", 0)) > 0
            ]
            actors[str(actor_id)] = {
                "actor_id": str(actor_id),
                "room_id": room,
                "episode": _episode_name(episode),
                "episode_dir": str(episode),
                "registered_value": value,
                "entity_kind": record.get("entity_kind"),
                "attribute_field": record.get("attribute_field") or record.get("appearance_field_used"),
                "stored_status": record.get("status"),
                "visible_pixel_frames": len(visible),
                "edge_frames": sum(
                    1 for frame in visible
                    if bbox_touches_frame_edge(frame.get("target_bbox_xyxy_px"), resolution_hw)
                ),
                "replayed_frames": 0,
                "accepted_frames": [],
                "status": "not_observable",
                "reasons": [],
                "cross_accepted": [],
                "cross_probed": [],
                "peak_visible_pixels": max((int(f.get("visible_pixels", 0)) for f in visible), default=0),
            }
            if value is None:
                actors[str(actor_id)]["status"] = "unregistered"
                continue
            for frame in visible:
                index = int(frame.get("frame_index", -1))
                if index in selected:
                    work.setdefault(index, []).append((str(actor_id), frame))
        probe_budget = Counter()
        for index in sorted(work):
            row = _mask_frame_row(index, frame_indices, modal.shape[0])
            rgb, _source = _rgb_frame(capture, index, frame_indices, rgb_array)
            if rgb.shape[:2] != modal.shape[1:]:
                raise ValueError(f"native RGB and modal mask resolutions differ: {episode}")
            modal_row = modal[row]
            for actor_id, frame in work[index]:
                state = actors[actor_id]
                instance = truth["per_instance"][actor_id]
                mask = modal_row == int(instance["semantic_id"])
                observed = inspect_registered_appearance(
                    rgb, mask, str(state["registered_value"]),
                    entity_kind=str(state["entity_kind"] or "human"),
                    target_bbox=frame.get("target_bbox_xyxy_px"),
                    **knobs,
                )
                state["replayed_frames"] += 1
                if observed.get("status") == "pass":
                    state["accepted_frames"].append(index)
                    state["status"] = "reviewed"
                elif observed.get("reason"):
                    state["reasons"].append(str(observed["reason"]))
                if probe_budget[actor_id] < int(cross_acceptance_frames):
                    probe_budget[actor_id] += 1
                    registered = str(state["registered_value"]).strip().casefold()
                    for candidate in CROSS_ACCEPTANCE_VALUES.get(str(state["entity_kind"]), ()):
                        if candidate == registered:
                            continue
                        probe = inspect_registered_appearance(
                            rgb, mask, candidate,
                            entity_kind=str(state["entity_kind"] or "human"),
                            target_bbox=frame.get("target_bbox_xyxy_px"),
                            **knobs,
                        )
                        state["cross_probed"].append(candidate)
                        if probe.get("status") == "pass":
                            state["cross_accepted"].append(candidate)
    for state in actors.values():
        state["cross_accepted"] = sorted(set(state["cross_accepted"]))
        state["cross_probed"] = sorted(set(state["cross_probed"]))
        state["reasons"] = [reason for reason, _ in Counter(state["reasons"]).most_common(3)]
    return {
        "schema": REPLAY_SCHEMA,
        "episode": _episode_name(episode),
        "episode_dir": str(episode),
        "room_id": room,
        "frame_stride": int(frame_stride),
        "reviewed_frame_count": len(selected),
        "appearance_review_path": str(review_path) if review_path else None,
        "actors": actors,
        "elapsed_s": round(time.monotonic() - started, 2),
    }


def replay_render_roots(
    roots: Iterable[Path],
    *,
    frame_stride: int = 10,
    cross_acceptance_frames: int = 2,
    exclude: Sequence[str] = (),
    thresholds: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Replay every episode under each render root, skipping excluded names."""
    blocked = tuple(str(value) for value in exclude)
    for root in roots:
        for episode in episode_directories(Path(root)):
            name = _episode_name(episode)
            if any(token in name or token in str(episode) for token in blocked):
                yield {"schema": REPLAY_SCHEMA, "episode": name, "episode_dir": str(episode),
                       "skipped": "excluded_by_request"}
                continue
            try:
                yield replay_episode(
                    episode,
                    frame_stride=frame_stride,
                    cross_acceptance_frames=cross_acceptance_frames,
                    thresholds=thresholds,
                )
            except (OSError, ValueError, KeyError) as error:
                yield {"schema": REPLAY_SCHEMA, "episode": name, "episode_dir": str(episode),
                       "failed": f"{type(error).__name__}: {error}"}


def confusion(
    episodes: Sequence[Mapping[str, Any]],
    *,
    minimum_visible_frames: int = 30,
) -> dict[str, Any]:
    """Registered value against verdict, per room and overall.

    The reported cohort is the one the change can actually act on: an actor with
    at least ``minimum_visible_frames`` frames carrying visible pixels and at
    least one of those frames whose bounding box does not touch the frame edge.
    Everything else is counted separately, because an actor the camera never
    really saw is a geometry outcome, not a classifier outcome.
    """
    rows: list[Mapping[str, Any]] = []
    for episode in episodes:
        if episode.get("skipped") or episode.get("failed"):
            continue
        rows.extend(episode.get("actors", {}).values())
    by_room: dict[str, Counter] = {}
    by_value: Counter = Counter()
    cohort = Counter()
    cross: Counter = Counter()
    by_split: dict[str, Counter] = {}
    cross_by_split: dict[str, Counter] = {}
    controlled_cross: list[dict[str, Any]] = []
    for row in rows:
        value = row.get("registered_value")
        if value is None:
            cohort["unregistered"] += 1
            continue
        readable = (
            int(row.get("visible_pixel_frames", 0)) >= int(minimum_visible_frames)
            and int(row.get("edge_frames", 0)) < int(row.get("visible_pixel_frames", 0))
        )
        cohort["readable" if readable else "not_readable"] += 1
        if not readable:
            continue
        room = str(row.get("room_id") or "unknown")
        status = str(row.get("status"))
        split = seed_group(str(row.get("episode")))
        by_room.setdefault(room, Counter())[status] += 1
        by_split.setdefault(split, Counter())[status] += 1
        by_value[(str(row.get("entity_kind")), str(value), status)] += 1
        if str(value) in CONTROLLED_TOP_COLORS:
            cross_by_split.setdefault(split, Counter())["controlled_actors"] += 1
        for candidate in row.get("cross_accepted", ()):
            cross[(str(value), str(candidate))] += 1
            if str(value) in CONTROLLED_TOP_COLORS and str(candidate) in CONTROLLED_TOP_COLORS:
                cross_by_split.setdefault(split, Counter())["controlled_cross_acceptances"] += 1
                controlled_cross.append({
                    "episode": row.get("episode"), "actor_id": row.get("actor_id"),
                    "seed_group": split,
                    "registered_value": value, "accepted_as": candidate,
                })
    return {
        "cohort": dict(cohort),
        "minimum_visible_frames": int(minimum_visible_frames),
        "by_room": {room: dict(counts) for room, counts in sorted(by_room.items())},
        "by_seed_group": {name: dict(counts) for name, counts in sorted(by_split.items())},
        "controlled_top_colors_by_seed_group": {
            name: dict(counts) for name, counts in sorted(cross_by_split.items())
        },
        "by_value": [
            {"entity_kind": kind, "registered_value": value, "status": status, "actors": count}
            for (kind, value, status), count in sorted(by_value.items())
        ],
        "cross_acceptance": [
            {"registered_value": value, "accepted_as": candidate, "actors": count}
            for (value, candidate), count in sorted(cross.items())
        ],
        "controlled_top_color_cross_acceptance": controlled_cross,
        "claim_boundary": (
            "a replay of retained pixels, not a re-render; the verdict is the same "
            "any-accepted-frame rule the delivery review applies"
        ),
    }
