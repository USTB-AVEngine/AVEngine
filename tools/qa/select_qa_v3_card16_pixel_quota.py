#!/usr/bin/env python3
"""Select card16 candidates after native-pixel truth, stratified by gold state.

Card16's final four-state answer does not exist until the native pixel join.
Geometry-time selection therefore cannot balance it without guessing.  This
tool consumes only pixel-qualified joins and declared room/candidate identity,
then round-robins rooms within each gold state.  It never reads model scores or
missing-modality probe outcomes.

Input JSON:
  {"records": [{"pilot_id": str, "room_id": str, "pixel_join": path}, ...]}

The output is research-candidate selection evidence.  A shortfall is reported,
not filled by relaxing pixel truth or inventing a room-specific trajectory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


STATES = (
    "visible_clear",
    "visible_occluded",
    "fully_occluded",
    "out_of_view",
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def _qualified(record):
    join_path = Path(record["pixel_join"]).resolve()
    join = _read(join_path)
    if join.get("profile_id") != "card16":
        raise ValueError(
            f"{record.get('pilot_id')}: pixel join is not card16")
    if join.get("status") != "pass":
        return None
    state = join.get("bindings", {}).get("main_truth_option")
    if state not in STATES:
        raise ValueError(
            f"{record.get('pilot_id')}: invalid card16 gold {state!r}")
    return {
        "pilot_id": str(record["pilot_id"]),
        "room_id": str(record["room_id"]),
        "pixel_join": str(join_path),
        "gold_state": state,
    }


def select(records, per_state):
    if per_state <= 0:
        raise ValueError("per_state must be positive")
    ids = [str(record.get("pilot_id")) for record in records]
    if any(value in {"", "None"} for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("pilot_id values must be non-empty and unique")

    qualified = []
    rejected_pixel = 0
    for record in records:
        item = _qualified(record)
        if item is None:
            rejected_pixel += 1
        else:
            qualified.append(item)

    by_state_room = {
        state: defaultdict(list) for state in STATES
    }
    for item in qualified:
        by_state_room[item["gold_state"]][item["room_id"]].append(item)
    for room_groups in by_state_room.values():
        for values in room_groups.values():
            values.sort(key=lambda item: item["pilot_id"])

    selected = []
    shortfall = {}
    for state in STATES:
        room_groups = by_state_room[state]
        rooms = sorted(room_groups)
        cursors = {room: 0 for room in rooms}
        while len([item for item in selected if item["gold_state"] == state]) < per_state:
            progressed = False
            for room in rooms:
                index = cursors[room]
                values = room_groups[room]
                if index >= len(values):
                    continue
                selected.append(values[index])
                cursors[room] += 1
                progressed = True
                if len([item for item in selected
                        if item["gold_state"] == state]) >= per_state:
                    break
            if not progressed:
                break
        got = sum(item["gold_state"] == state for item in selected)
        shortfall[state] = max(0, per_state - got)

    selected_ids = {item["pilot_id"] for item in selected}
    for item in qualified:
        item["selected"] = item["pilot_id"] in selected_ids
    selected_counts = Counter(item["gold_state"] for item in selected)
    room_counts = Counter(item["room_id"] for item in selected)
    return {
        "schema": "qa_v3_card16_pixel_quota_selection_v1",
        "status": "complete" if not any(shortfall.values()) else "partial",
        "qualification_claim": False,
        "selection_authority": (
            "native-pixel gold state only; no model or probe outcome read"),
        "requested_per_state": per_state,
        "input_count": len(records),
        "pixel_qualified_count": len(qualified),
        "pixel_rejected_count": rejected_pixel,
        "selected_count": len(selected),
        "selected_by_gold": {
            state: selected_counts.get(state, 0) for state in STATES},
        "selected_by_room": dict(sorted(room_counts.items())),
        "shortfall_by_gold": shortfall,
        "records": sorted(qualified, key=lambda item: item["pilot_id"]),
        "boundary": (
            "Post-pixel research selection only; does not establish modality "
            "resistance, human answerability, or dataset admission."),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--per-state", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        print(f"refusing to overwrite: {args.output}", file=sys.stderr)
        return 2
    index = _read(args.index)
    result = select(index.get("records", []), args.per_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, result)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": result["status"],
        "selected_by_gold": result["selected_by_gold"],
        "shortfall_by_gold": result["shortfall_by_gold"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
