"""Write neutral capture records from actual Habitat root, emitter and sensor readback."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from avengine.capture.neutral_readback import (
    COORDINATE_FRAME, SCHEMA, observed_motion, validate_clock, write_neutral_readback,
)
from avengine.capture.orientation import habitat_basis_from_xyzw


def neutral_from_habitat_readbacks(records: Mapping[str, Any], roots: np.ndarray,
                                   emitters: np.ndarray, plan: Mapping[str, Any], *,
                                   source_readbacks: list[str],
                                   actor_from_skin_root_by_slot: Mapping[str, Any] | None = None) -> dict:
    clock = validate_clock(plan["clock"])
    observed_clock = records["render"]
    for key in ("frame_count", "frame_rate_hz", "sample_rate_hz", "sample_count",
                "time_base_hz", "ticks_per_frame"):
        if observed_clock.get(key) != clock[key]:
            raise ValueError(f"Habitat observed clock differs from plan: {key}")
    frames = records["frames"]
    if not frames:
        raise ValueError("Habitat readback has no frames")
    first = frames[0]["actor_readbacks"]
    slots = [actor["source_slot_id"] for actor in first]
    if len(set(slots)) != len(slots):
        raise ValueError("Habitat source slots must be unique")
    count, actors = int(clock["frame_count"]), len(slots)
    if roots.shape != (count, actors, 4, 4) or emitters.shape != (count, actors, 3):
        raise ValueError("Habitat root/emitter arrays do not match the plan clock and entities")
    if not np.isfinite(roots).all() or not np.isfinite(emitters).all():
        raise ValueError("Habitat readback arrays contain nonfinite values")
    canonical_roots = roots.copy()
    root_frames = {}
    for j, slot in enumerate(slots):
        transform = (actor_from_skin_root_by_slot or {}).get(slot)
        if transform is None:
            root_frames[slot] = "native_skin_or_object_root"
            continue
        transform = np.asarray(transform, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError(f"Habitat asset root transform is invalid: {slot}")
        canonical_roots[:, j] = roots[:, j] @ np.linalg.inv(transform)
        root_frames[slot] = "asset_actor_root_from_observed_skin_and_package_transform"
    moving = {slot: observed_motion(canonical_roots[:, j, :3, 3], clock["frame_rate_hz"])
              for j, slot in enumerate(slots)}
    camera, entities = [], {slot: [] for slot in slots}
    for frame in frames:
        i = frame["frame_index"]
        if i != len(camera) or i >= count or frame["pts_ticks"] != i * clock["ticks_per_frame"]:
            raise ValueError("Habitat frame identity or clock drift")
        modalities = frame["modalities"]
        rgb_sensor = modalities["rgb"]["sensor_uuid"]
        sensors = frame["camera_readback"]["sensors"]
        sensor = sensors[rgb_sensor]
        basis = habitat_basis_from_xyzw(sensor["rotation_xyzw"])
        # RGB, depth and semantic are one co-located camera/listener rig.
        for spec in modalities.values():
            other = sensors[spec["sensor_uuid"]]
            if not np.allclose(other["translation_m"], sensor["translation_m"], atol=1e-6, rtol=0):
                raise ValueError("Habitat sensors are not co-located")
            other_basis = habitat_basis_from_xyzw(other["rotation_xyzw"])
            if not np.allclose([other_basis.forward_xyz, other_basis.up_xyz],
                               [basis.forward_xyz, basis.up_xyz], atol=1e-6, rtol=0):
                raise ValueError("Habitat sensors have different orientations")
        camera.append({"frame_index": i, "pts_ticks": frame["pts_ticks"],
                       "position_m": sensor["translation_m"],
                       "basis": {"forward": list(basis.forward_xyz),
                                 "right": list(basis.right_xyz), "up": list(basis.up_xyz)}})
        actor_records = frame["actor_readbacks"]
        if [x["source_slot_id"] for x in actor_records] != slots:
            raise ValueError("Habitat actor order or source identity drift")
        for j, (slot, actor) in enumerate(zip(slots, actor_records, strict=True)):
            if actor["actor_id"] != first[j]["actor_id"] or actor["asset_id"] != first[j]["asset_id"]:
                raise ValueError("Habitat actor/asset identity drift")
            if not np.allclose(actor["world_from_skin_root"], roots[i, j], atol=1e-6, rtol=0):
                raise ValueError("Habitat root JSON and array readbacks disagree")
            if not np.allclose(actor["emitter_world_position_m"], emitters[i, j], atol=1e-6, rtol=0):
                raise ValueError("Habitat emitter JSON and array readbacks disagree")
            entities[slot].append({"frame_index": i, "pts_ticks": frame["pts_ticks"],
                                   "root": canonical_roots[i, j, :3, 3].tolist(),
                                   "emitter": emitters[i, j].tolist(), "moving": moving[slot][i]})
    return {"schema": SCHEMA, "clock": clock, "coordinate_frame": dict(COORDINATE_FRAME),
            "camera": camera, "entities": entities,
            "entity_identities": {a["source_slot_id"]: {k: a[k] for k in
                                  ("actor_id", "asset_id", "source_endpoint_id")} for a in first},
            "producer": {"module": __name__, "renderer": "habitat",
                         "source_readbacks": source_readbacks,
                         "world_transform": "identity_meter_y_up_right",
                         "root_frames_by_entity": root_frames,
                         "movement": "observed_root_forward_difference_gt_0.05_mps; last repeats previous"}}


def write_habitat_neutral_readback(capture_dir: Path, plan: Mapping[str, Any],
                                   output_path: Path | None = None) -> dict:
    root = Path(capture_dir)
    sources = [root / name for name in
               ("frame_records.json", "actor_root_readbacks.npy", "emitter_positions_m.npy")]
    transforms = {}
    receipt_path = root / "research_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        case_path = Path(receipt["inputs"]["case_manifest"])
        case = json.loads(case_path.read_text())
        sources.append(case_path)
        for record in case["actor_tracks"]:
            track_path = Path(record["track_path"])
            if not track_path.is_absolute():
                track_path = case_path.parent / track_path
            track = json.loads(track_path.read_text())
            sources.append(track_path)
            transform = track["asset"].get("actor_from_skin_root")
            if transform is not None:
                transforms[track["source_slot_id"]] = transform
            elif track.get("entity_class") in {"rigid_object", "rigid_static_object"} or track["asset"].get("entity_class") in {"rigid_object", "rigid_static_object"}:
                transforms[track["source_slot_id"]] = np.eye(4).tolist()
            elif plan.get("plan_coordinates") == "renderer_neutral":
                raise ValueError(f"common Habitat track has no package actor/skin root transform: {track_path}")
    elif plan.get("plan_coordinates") == "renderer_neutral":
        raise ValueError("common Habitat readback needs its native capture receipt")
    data = neutral_from_habitat_readbacks(
        json.loads(sources[0].read_text()), np.load(sources[1], allow_pickle=False),
        np.load(sources[2], allow_pickle=False), plan,
        source_readbacks=[str(path.resolve()) for path in sources],
        actor_from_skin_root_by_slot=transforms)
    write_neutral_readback(output_path or root / "neutral_readback.json", data, plan=plan)
    return data
