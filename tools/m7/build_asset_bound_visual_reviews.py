#!/usr/bin/env python3
"""Build Habitat-only internal visual QA reviews for the M7 throughput batch.

The 1,000-item batch intentionally has no duplicated RGB or Topdown media.
This research-only tool renders a tiny, explicitly selected review subset.  It
will only mux the particular ``v00`` mixture whose episode, asset binding,
root paths, source-center paths, and captured actor transforms agree.  It is
not an Apartment final-visual renderer: its left panel is Habitat internal QA;
the SPEAR/UE optional backend owns an Apartment presentation RGB frame.  The
right panel is diagnostic QA only, never a dataset camera view.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import sha256_file, write_json
from avengine.m5_1.review import (
    SourceOverlayTrack,
    compose_annotated_frames,
    encode_annotated_review,
)
from avengine.m6x.geometry import RuntimeObstacleMap
from avengine.m6x.topdown import render_runtime_topdown_frames


FRAME_COUNT = 75
FRAME_RATE_HZ = 15
SOURCE_SLOTS = ("source1", "source2")
LISTENER_POSITION_M = (-0.7, 1.471, 0.65)
LISTENER_YAW_DEG = 55.0
CAMERA_HFOV_DEG = 105.0
REVIEW_SCHEMA = "avengine_m7_asset_bound_habitat_qa_review_v1"


class AssetBoundReviewError(RuntimeError):
    """A requested review would not faithfully represent one batch item."""


@dataclass(frozen=True)
class ReviewSpec:
    """One explicitly bound visual/audio item selected for human listening."""

    episode_id: str
    capture_directory: str
    sample_id: str
    labels: tuple[str, str]
    asset_ids: tuple[str, str]
    actor_classes: tuple[str, str]
    asset_classes: tuple[str, str]
    sound_classes: tuple[str, str]
    colors: tuple[tuple[int, int, int], tuple[int, int, int]]
    local_forward_axes: tuple[tuple[float, float, float], tuple[float, float, float]]
    emitter_offsets_m: tuple[tuple[float, float, float], tuple[float, float, float]]


REVIEW_SPECS = (
    ReviewSpec(
        episode_id="human_cat__both_moving_000",
        capture_directory="human_cat__both_moving_000_r2",
        sample_id="human_cat__both_moving_000__v00",
        labels=("Human", "Cat"),
        asset_ids=(
            "rocketbox_human_male_adult_01_m5_1_candidate",
            "quaternius_domestic_cat_generic_diagnostic_v1",
        ),
        actor_classes=("human", "cat"),
        asset_classes=("rocketbox_human", "quaternius_cat"),
        sound_classes=("human_vocalization", "cat_vocalization"),
        colors=((42, 210, 220), (250, 120, 70)),
        local_forward_axes=((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        emitter_offsets_m=((0.0, 1.61, 0.0), (0.312, 0.252, 0.0)),
    ),
    ReviewSpec(
        episode_id="small_dog_cat__both_moving_047",
        capture_directory="small_dog_cat__both_moving_047_r4",
        sample_id="small_dog_cat__both_moving_047__v00",
        labels=("Beagle", "Cat"),
        asset_ids=(
            "rocketbox_dog_beagle_01_m2_v7_world_contact_candidate",
            "quaternius_domestic_cat_generic_diagnostic_v1",
        ),
        actor_classes=("beagle", "cat"),
        asset_classes=("rocketbox_beagle", "quaternius_cat"),
        sound_classes=("dog_vocalization", "cat_vocalization"),
        colors=((42, 210, 220), (250, 120, 70)),
        local_forward_axes=((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        emitter_offsets_m=((0.424, 0.407, -0.137), (0.312, 0.252, 0.0)),
    ),
)


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetBoundReviewError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AssetBoundReviewError(f"JSON object required: {path}")
    return value


def _require_regular_file(path: Path, *, owner: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise AssetBoundReviewError(f"{owner} must be a regular file: {path}")


def _load_bank(plan_root: Path) -> tuple[Mapping[str, Any], Mapping[str, int], np.lib.npyio.NpzFile]:
    record = _load_json(plan_root / "trajectory_bank.json")
    if record.get("frame_count") != FRAME_COUNT or record.get("frame_rate_hz") != FRAME_RATE_HZ:
        raise AssetBoundReviewError("trajectory bank has an unexpected M7 clock")
    arrays = np.load(plan_root / "trajectory_bank.npz", allow_pickle=False)
    needed = {
        "episode_ids",
        "source_slot_ids",
        "source_root_paths_m",
        "source_center_paths_m",
    }
    if not needed.issubset(set(arrays.files)):
        raise AssetBoundReviewError("trajectory bank arrays are incomplete")
    source_slots = tuple(str(value) for value in arrays["source_slot_ids"])
    if source_slots != SOURCE_SLOTS:
        raise AssetBoundReviewError("trajectory bank source-slot order differs")
    ids = tuple(str(value) for value in arrays["episode_ids"])
    if len(ids) != len(set(ids)):
        raise AssetBoundReviewError("trajectory bank repeats an episode ID")
    index = {episode_id: ordinal for ordinal, episode_id in enumerate(ids)}
    roots = arrays["source_root_paths_m"]
    centers = arrays["source_center_paths_m"]
    expected_shape = (len(ids), 2, FRAME_COUNT, 3)
    if roots.shape != expected_shape or centers.shape != expected_shape:
        raise AssetBoundReviewError("trajectory bank path arrays have an unexpected shape")
    if not np.all(np.isfinite(roots)) or not np.all(np.isfinite(centers)):
        raise AssetBoundReviewError("trajectory bank paths are not finite")
    return record, index, arrays


def _obstacle_map(bank_root: Path) -> RuntimeObstacleMap:
    feasible = _load_json(bank_root / "feasible_region.json")
    authority = feasible.get("obstacle_authority")
    if not isinstance(authority, Mapping):
        raise AssetBoundReviewError("feasible region lacks obstacle authority")
    mask_path = bank_root / "feasible_region_source1.npz"
    _require_regular_file(mask_path, owner="source1 feasibility map")
    loaded = np.load(mask_path, allow_pickle=False)
    if "navmesh_mask" not in loaded.files:
        raise AssetBoundReviewError("source1 feasibility map lacks navmesh_mask")
    navmesh = np.asarray(loaded["navmesh_mask"], dtype=np.uint8)
    bounds = np.asarray(authority.get("bounds_m"), dtype=np.float64)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise AssetBoundReviewError("obstacle authority bounds are invalid")
    return RuntimeObstacleMap(
        binary_navmesh=np.ascontiguousarray(navmesh),
        bounds_m=tuple(tuple(float(value) for value in row) for row in bounds),
        floor_height_m=float(authority["floor_height_m"]),
        meters_per_pixel=float(authority["meters_per_pixel"]),
        rigid_obstacles=tuple(authority.get("rigid_obstacles", ())),
        authority=str(authority["authority"]),
        claim_boundary=str(authority["claim_boundary"]),
        rigid_obstacles_baked_into_navmesh=bool(
            authority.get("rigid_obstacles_baked_into_navmesh", False)
        ),
    )


def _world_centers(world_matrices: np.ndarray, offsets: Sequence[Sequence[float]]) -> np.ndarray:
    matrices = np.asarray(world_matrices, dtype=np.float64)
    if matrices.shape != (FRAME_COUNT, 2, 4, 4) or not np.all(np.isfinite(matrices)):
        raise AssetBoundReviewError("captured actor transforms must be finite [75,2,4,4]")
    result = np.empty((FRAME_COUNT, 2, 3), dtype=np.float64)
    for actor_index, offset in enumerate(offsets):
        vector = np.asarray(offset, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise AssetBoundReviewError("emitter offsets must be finite vec3")
        homogeneous = np.concatenate((vector, np.asarray([1.0], dtype=np.float64)))
        result[:, actor_index] = np.einsum(
            "nij,j->ni", matrices[:, actor_index], homogeneous
        )[:, :3]
    return np.ascontiguousarray(result)


def _heading_xz(world_matrices: np.ndarray, axes: Sequence[Sequence[float]]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for actor_index, axis in enumerate(axes):
        local = np.asarray(axis, dtype=np.float64)
        if local.shape != (3,) or not np.all(np.isfinite(local)):
            raise AssetBoundReviewError("local forward axes must be finite vec3")
        directions = np.einsum(
            "nij,j->ni", world_matrices[:, actor_index, :3, :3], local
        )[:, (0, 2)]
        if np.any(np.linalg.norm(directions, axis=1) <= 1.0e-12):
            raise AssetBoundReviewError("captured actor has a zero horizontal heading")
        result[SOURCE_SLOTS[actor_index]] = np.ascontiguousarray(directions)
    return result


def _sample_record(batch_root: Path, sample_id: str) -> Mapping[str, Any]:
    record = _load_json(batch_root / "samples.json")
    rows = record.get("samples")
    if not isinstance(rows, list):
        raise AssetBoundReviewError("batch samples record is malformed")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("sample_id") == sample_id]
    if len(matches) != 1:
        raise AssetBoundReviewError(f"batch sample {sample_id!r} is not unique")
    row = matches[0]
    if row.get("both_sources_active") is not True:
        raise AssetBoundReviewError("review sample does not have both sources active")
    return row


def _asset_bindings(plan_root: Path) -> dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]]:
    record = _load_json(plan_root / "asset_emitter_binding_report.json")
    rows = record.get("scenarios")
    if record.get("status") != "pass" or not isinstance(rows, list):
        raise AssetBoundReviewError("asset-emitter binding report is malformed")
    result: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for row in rows:
        binding_report = row.get("binding_report") if isinstance(row, Mapping) else None
        bindings = binding_report.get("bindings") if isinstance(binding_report, Mapping) else None
        episode_id = row.get("output_episode_id") if isinstance(row, Mapping) else None
        if not isinstance(episode_id, str) or not isinstance(bindings, list) or len(bindings) != 2:
            raise AssetBoundReviewError("asset-emitter binding scenario is malformed")
        by_slot = {
            binding.get("source_slot_id"): binding
            for binding in bindings
            if isinstance(binding, Mapping)
        }
        if set(by_slot) != set(SOURCE_SLOTS):
            raise AssetBoundReviewError("asset-emitter binding source slots differ")
        result[episode_id] = (by_slot["source1"], by_slot["source2"])
    return result


def _center_gate(gate: Mapping[str, Any], episode_id: str) -> tuple[bool, dict[str, float]]:
    sources = gate.get("sources")
    if not isinstance(sources, Mapping):
        raise AssetBoundReviewError("navmesh center gate is malformed")
    clearances: dict[str, float] = {}
    for slot in SOURCE_SLOTS:
        row = sources.get(f"{episode_id}::{slot}")
        if not isinstance(row, Mapping) or row.get("status") != "pass":
            raise AssetBoundReviewError(f"center gate fails for {episode_id} {slot}")
        clearance = float(row.get("minimum_navmesh_clearance_m"))
        if not np.isfinite(clearance) or clearance < 0.0:
            raise AssetBoundReviewError("center gate clearance is invalid")
        clearances[slot] = clearance
    return True, clearances


def _capture_arrays(
    capture_root: Path,
    *,
    expected_actor_classes: Sequence[str],
    expected_asset_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    evidence = _load_json(capture_root / "evidence.json")
    if evidence.get("status") != "pass" or evidence.get("research_only") is not True:
        raise AssetBoundReviewError("review capture is not a successful research capture")
    actors = evidence.get("actors")
    classes = (
        tuple(str(row.get("actor_class")) for row in actors)
        if isinstance(actors, list)
        else ()
    )
    asset_ids = (
        tuple(str(row.get("asset_id")) for row in actors)
        if isinstance(actors, list)
        else ()
    )
    if classes != tuple(expected_actor_classes):
        raise AssetBoundReviewError("review capture actor classes differ from its requested asset pair")
    if asset_ids != tuple(expected_asset_ids):
        raise AssetBoundReviewError("review capture asset IDs differ from its requested asset pair")
    rgb_path = capture_root / "arrays" / "rgb.npy"
    matrix_path = capture_root / "arrays" / "actor_world_matrices.npy"
    _require_regular_file(rgb_path, owner="captured RGB")
    _require_regular_file(matrix_path, owner="captured actor transforms")
    rgb = np.load(rgb_path, allow_pickle=False)
    matrices = np.load(matrix_path, allow_pickle=False)
    if rgb.shape != (FRAME_COUNT, 240, 320, 3) or rgb.dtype != np.uint8:
        raise AssetBoundReviewError("captured RGB has an unexpected shape")
    return np.ascontiguousarray(rgb), np.ascontiguousarray(matrices, dtype=np.float64)


def _review_one(
    *,
    spec: ReviewSpec,
    plan_root: Path,
    bank_root: Path,
    batch_root: Path,
    capture_root: Path,
    output_root: Path,
    bank_index: Mapping[str, int],
    bank_arrays: np.lib.npyio.NpzFile,
    obstacle_map: RuntimeObstacleMap,
    center_gate: Mapping[str, Any],
    asset_bindings: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> Mapping[str, Any]:
    if spec.episode_id not in bank_index:
        raise AssetBoundReviewError(f"review episode missing from plan: {spec.episode_id}")
    sample = _sample_record(batch_root, spec.sample_id)
    if sample.get("episode_id") != spec.episode_id or sample.get("variant_index") != 0:
        raise AssetBoundReviewError("review sample does not bind the requested v00 episode")
    assets = sample.get("asset_ids_by_source_slot")
    expected_assets = {slot: spec.asset_ids[index] for index, slot in enumerate(SOURCE_SLOTS)}
    if not isinstance(assets, Mapping) or dict(assets) != expected_assets:
        raise AssetBoundReviewError("review sample asset IDs differ from the requested visual pair")
    bindings = asset_bindings.get(spec.episode_id)
    if bindings is None:
        raise AssetBoundReviewError("review episode lacks an asset-emitter binding")
    for source_index, binding in enumerate(bindings):
        if binding.get("asset_id") != spec.asset_ids[source_index]:
            raise AssetBoundReviewError("review binding asset differs from the requested visual pair")
        if not np.array_equal(
            np.asarray(binding.get("emitter_offset_m"), dtype=np.float64),
            np.asarray(spec.emitter_offsets_m[source_index], dtype=np.float64),
        ) or not np.array_equal(
            np.asarray(binding.get("local_anatomical_forward_axis"), dtype=np.float64),
            np.asarray(spec.local_forward_axes[source_index], dtype=np.float64),
        ):
            raise AssetBoundReviewError("review binding axis or emitter offset differs from the visual spec")
    audio = sample.get("audio")
    if not isinstance(audio, Mapping) or audio.get("sample_rate_hz") != 16000 or audio.get("sample_count") != 80000:
        raise AssetBoundReviewError("review sample has an unexpected audio contract")
    mixture = audio.get("mixture")
    if not isinstance(mixture, Mapping) or not isinstance(mixture.get("path"), str):
        raise AssetBoundReviewError("review sample lacks a mixture path")
    mixture_path = batch_root / "audio" / "binaural" / str(mixture["path"])
    _require_regular_file(mixture_path, owner="review mixture")
    expected_hash = mixture.get("audio_sha256")
    if not isinstance(expected_hash, str) or sha256_file(mixture_path) != expected_hash:
        raise AssetBoundReviewError("review mixture no longer matches the batch sidecar")

    capture = capture_root / spec.capture_directory
    rgb, matrices = _capture_arrays(
        capture,
        expected_actor_classes=spec.actor_classes,
        expected_asset_ids=spec.asset_ids,
    )
    expected_centers = np.ascontiguousarray(
        bank_arrays["source_center_paths_m"][bank_index[spec.episode_id]].transpose(1, 0, 2)
    )
    reconstructed_centers = _world_centers(matrices, spec.emitter_offsets_m)
    center_error = float(np.max(np.abs(expected_centers - reconstructed_centers)))
    if center_error > 1.0e-9:
        raise AssetBoundReviewError(
            f"capture source centers differ from asset-bound plan by {center_error:.3g} m"
        )
    pass_gate, clearances = _center_gate(center_gate, spec.episode_id)
    headings = _heading_xz(matrices, spec.local_forward_axes)
    center_paths = {
        slot: expected_centers[:, source_index]
        for source_index, slot in enumerate(SOURCE_SLOTS)
    }
    topdown = render_runtime_topdown_frames(
        obstacle_map,
        center_paths,
        listener_position_m=LISTENER_POSITION_M,
        listener_yaw_deg=LISTENER_YAW_DEG,
        camera_hfov_degrees=CAMERA_HFOV_DEG,
        source_activity_by_frame={slot: np.ones(FRAME_COUNT, dtype=np.bool_) for slot in SOURCE_SLOTS},
        source_heading_xz_by_frame=headings,
        source_labels={slot: spec.labels[index] for index, slot in enumerate(SOURCE_SLOTS)},
        source_colors={slot: spec.colors[index] for index, slot in enumerate(SOURCE_SLOTS)},
    )
    tracks = tuple(
        SourceOverlayTrack(
            source_id=slot,
            label=spec.labels[index],
            asset_class=spec.asset_classes[index],
            sound_class=spec.sound_classes[index],
            color_rgb=spec.colors[index],
            positions_m=center_paths[slot],
            current_event_by_frame=("simultaneous_vocalization",) * FRAME_COUNT,
            active_by_frame=(True,) * FRAME_COUNT,
            true_flags=("simultaneous", "moving"),
            center_clearance_m=np.full(FRAME_COUNT, clearances[slot], dtype=np.float64),
        )
        for index, slot in enumerate(SOURCE_SLOTS)
    )
    frames = compose_annotated_frames(
        main_rgb=rgb,
        topdown_rgb=topdown,
        tracks=tracks,
        clip_id=spec.sample_id,
        room_id="spear_apartment_0000",
        review_stage_label="M7 internal Habitat QA; not SPEAR final RGB",
        listener_position_m=LISTENER_POSITION_M,
        listener_yaw_deg=LISTENER_YAW_DEG,
        aggregate_true_flags=("both_sources_active", "asset_bound_rir"),
        audio_diagnostic_by_frame=(
            "exact v00 native RLR-HRTF binaural mixture; audio remains 360 degrees",
        ) * FRAME_COUNT,
        center_gate_pass=pass_gate,
        fps=FRAME_RATE_HZ,
    )
    destination = output_root / f"{spec.sample_id}_rgb_topdown_binaural.mp4"
    media = encode_annotated_review(
        frames,
        destination,
        fps=FRAME_RATE_HZ,
        audio_path=mixture_path,
    )
    return {
        "episode_id": spec.episode_id,
        "sample_id": spec.sample_id,
        "capture_directory": str(capture),
        "asset_ids_by_source_slot": expected_assets,
        "mixture_wav": {
            "path": str(mixture_path),
            "sha256": expected_hash,
            "exact_batch_sample": True,
        },
        "asset_bound_center_reconstruction_maximum_error_m": center_error,
        "center_gate": {"status": "pass", "minimum_navmesh_clearance_m": clearances},
        "topdown": {
            "qa_only": True,
            "source_centers_from": "asset-bound plan source_center_paths_m",
            "headings_from": "capture actor_world_matrices x asset local forward axes",
        },
        "media": media,
    }


def build_reviews(
    *,
    plan_root: Path,
    feasibility_bank_root: Path,
    batch_root: Path,
    capture_root: Path,
    output_root: Path,
) -> Path:
    """Build the two explicit research listening videos without overwriting."""

    output_root = output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise AssetBoundReviewError(f"refusing to replace review output: {output_root}")
    plan_root = plan_root.resolve()
    bank_root = feasibility_bank_root.resolve()
    batch_root = batch_root.resolve()
    capture_root = capture_root.resolve()
    _bank_record, bank_index, bank_arrays = _load_bank(plan_root)
    gate = _load_json(plan_root / "navmesh_center_gate.json")
    if gate.get("status") != "pass":
        raise AssetBoundReviewError("asset-bound center gate did not pass")
    obstacle_map = _obstacle_map(bank_root)
    asset_bindings = _asset_bindings(plan_root)
    output_root.mkdir(parents=True)
    try:
        entries = [
            _review_one(
                spec=spec,
                plan_root=plan_root,
                bank_root=bank_root,
                batch_root=batch_root,
                capture_root=capture_root,
                output_root=output_root,
                bank_index=bank_index,
                bank_arrays=bank_arrays,
                obstacle_map=obstacle_map,
                center_gate=gate,
                asset_bindings=asset_bindings,
            )
            for spec in REVIEW_SPECS
        ]
        receipt = {
            "schema": REVIEW_SCHEMA,
            "status": "pass",
            "research_only": True,
            "qualification_claim": False,
            "claim_boundary": "two exact asset-bound M7 Habitat internal QA reviews; they are not SPEAR/UE final RGB, a dataset view, or dataset admission",
            "render_backend": "Habitat_internal_QA_only",
            "frame_count": FRAME_COUNT,
            "frame_rate_hz": FRAME_RATE_HZ,
            "plan_root": str(plan_root),
            "feasibility_bank_root": str(bank_root),
            "batch_root": str(batch_root),
            "entries": entries,
        }
        write_json(output_root / "review_receipt.json", receipt)
        return output_root
    except BaseException:
        # Keep successfully written evidence for inspection, but never write a
        # pass receipt for a partial review set.
        raise
    finally:
        bank_arrays.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--feasibility-bank-root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_reviews(
        plan_root=args.plan_root,
        feasibility_bank_root=args.feasibility_bank_root,
        batch_root=args.batch_root,
        capture_root=args.capture_root,
        output_root=args.output,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
