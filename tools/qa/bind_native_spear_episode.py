#!/usr/bin/env python3
"""Bind one compiled QA Fact table to retained native SPEAR/UE evidence.

This tool does not render or infer missing pixels.  It proves that the Facts
and QuestionSpecs describe the exact retained episode whose camera/source
poses, RGB frames and binaural packets passed the native runtime readback.  An
optional full native-pixel manifest additionally binds same-camera normal and
show-only metric depth, depth-derived modal masks and their compiled Facts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.contracts.json_io import canonical_json_sha256, write_json  # noqa: E402
from avengine.dataset.sensor_rig import (  # noqa: E402
    m7_sensor_rig_binding,
    m7_sensor_rig_pose_series,
)
from avengine.qa.question_spec import evaluate_question_specs  # noqa: E402


SCHEMA = "avengine_qa_native_spear_episode_binding_v1"
POSITION_TOLERANCE_CM = 1.0e-6
ANGLE_TOLERANCE_DEG = 1.0e-6


class NativeEpisodeBindingError(RuntimeError):
    """Retained native evidence does not match the requested QA Episode."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NativeEpisodeBindingError(message)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NativeEpisodeBindingError(f"cannot read JSON {path}: {error}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_inventory(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": str(file.relative_to(path)),
            "size_bytes": file.stat().st_size,
            "sha256": _sha256_file(file),
        }
        for file in sorted(item for item in path.rglob("*") if item.is_file())
    ]


def _file_record(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"retained artifact is missing: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _audio_packet_sha256(path: Path) -> str:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-c",
            "copy",
            "-f",
            "streamhash",
            "-hash",
            "sha256",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    prefix = "0,a,SHA256="
    _require(
        len(lines) == 1 and lines[0].startswith(prefix),
        f"unexpected audio streamhash for {path}: {lines}",
    )
    return lines[0][len(prefix) :].lower()


def _probe_media(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    video = [item for item in payload["streams"] if item["codec_type"] == "video"]
    audio = [item for item in payload["streams"] if item["codec_type"] == "audio"]
    _require(len(video) == 1 and len(audio) == 1, "native review needs one AV stream")
    return {
        "video_codec": video[0].get("codec_name"),
        "width": int(video[0]["width"]),
        "height": int(video[0]["height"]),
        "frame_rate": video[0].get("avg_frame_rate"),
        "frame_count": int(video[0]["nb_read_frames"]),
        "audio_codec": audio[0].get("codec_name"),
        "audio_channels": int(audio[0]["channels"]),
        "audio_sample_rate_hz": int(audio[0]["sample_rate"]),
        "duration_seconds": float(payload["format"]["duration"]),
        "audio_packet_sha256": _audio_packet_sha256(path),
    }


def _angle_error_deg(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _world_position_to_ue_cm(position_m: Sequence[float]) -> np.ndarray:
    position = np.asarray(position_m, dtype=np.float64)
    _require(position.shape == (3,), "world position must be xyz")
    return 100.0 * position[[0, 2, 1]]


def _world_yaw_to_ue_deg(yaw_deg: float) -> float:
    return (-float(yaw_deg) - 90.0 + 180.0) % 360.0 - 180.0


def _check_runtime_alignment(
    *,
    facts: Mapping[str, Any],
    sensor_rig: Mapping[str, Any],
    runtime_readbacks: Mapping[str, Any],
) -> dict[str, Any]:
    frame_count = int(facts["time"]["frame_count"])
    series = m7_sensor_rig_pose_series(sensor_rig)
    _require(len(series.pose_hashes) == frame_count, "SensorRig frame count drift")
    listener = facts["listener"]
    fact_positions = np.asarray(listener["positions_m_by_frame"], dtype=np.float64)
    fact_orientations = np.asarray(
        listener["orientations_wxyz_by_frame"], dtype=np.float64
    )
    fact_yaws = np.asarray(listener["yaw_deg_by_frame"], dtype=np.float64)
    _require(
        fact_positions.shape == series.positions_m.shape
        and np.allclose(fact_positions, series.positions_m, rtol=0.0, atol=1.0e-12),
        "Facts Listener positions differ from SensorRigTrajectory",
    )
    _require(
        fact_orientations.shape == series.orientations_wxyz.shape
        and np.allclose(
            np.abs(np.sum(fact_orientations * series.orientations_wxyz, axis=1)),
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        ),
        "Facts Listener orientations differ from SensorRigTrajectory",
    )
    _require(
        fact_yaws.shape == series.yaws_deg.shape
        and all(
            _angle_error_deg(float(left), float(right)) <= 1.0e-9
            for left, right in zip(fact_yaws, series.yaws_deg)
        ),
        "Facts Listener yaws differ from SensorRigTrajectory",
    )

    camera = runtime_readbacks.get("camera_root")
    _require(isinstance(camera, list) and len(camera) == frame_count, "camera readback is incomplete")
    maximum_camera_position_error_cm = 0.0
    maximum_camera_yaw_error_deg = 0.0
    for index, record in enumerate(camera):
        _require(record.get("frame_index") == index, "camera readback frame order drift")
        _require(
            record.get("expected_pose_hash") == series.pose_hashes[index],
            f"camera pose hash drift at frame {index}",
        )
        observed = np.asarray(record["location_cm"], dtype=np.float64)
        expected = _world_position_to_ue_cm(fact_positions[index])
        position_error = float(np.max(np.abs(observed - expected)))
        yaw_error = _angle_error_deg(
            float(record["rotation_deg"][2]),
            _world_yaw_to_ue_deg(float(fact_yaws[index])),
        )
        maximum_camera_position_error_cm = max(
            maximum_camera_position_error_cm, position_error
        )
        maximum_camera_yaw_error_deg = max(maximum_camera_yaw_error_deg, yaw_error)
    _require(
        maximum_camera_position_error_cm <= POSITION_TOLERANCE_CM,
        "native camera locations differ from Facts",
    )
    _require(
        maximum_camera_yaw_error_deg <= ANGLE_TOLERANCE_DEG,
        "native camera rotations differ from Facts",
    )

    actor_metrics: dict[str, Any] = {}
    actor_readbacks = runtime_readbacks.get("actor_roots")
    _require(isinstance(actor_readbacks, Mapping), "actor root readback is missing")
    for slot_id, track in facts["tracks"]["instances"].items():
        actor_id = f"{slot_id}_actor"
        records = actor_readbacks.get(actor_id)
        _require(
            isinstance(records, list) and len(records) == frame_count,
            f"native actor readback is incomplete for {actor_id}",
        )
        roots = track["root_position_m"]
        yaws = track["facing_yaw_deg"]
        _require(isinstance(yaws, list), f"Facts facing yaw is missing for {slot_id}")
        max_position = 0.0
        max_yaw = 0.0
        for index, record in enumerate(records):
            _require(record.get("frame_index") == index, f"{actor_id} frame order drift")
            observed = np.asarray(record["location_cm"], dtype=np.float64)
            expected = _world_position_to_ue_cm(roots[index])
            max_position = max(max_position, float(np.max(np.abs(observed - expected))))
            max_yaw = max(
                max_yaw,
                _angle_error_deg(
                    float(record["rotation_deg"][2]),
                    _world_yaw_to_ue_deg(float(yaws[index])),
                ),
            )
        _require(max_position <= POSITION_TOLERANCE_CM, f"{actor_id} position drift")
        _require(max_yaw <= ANGLE_TOLERANCE_DEG, f"{actor_id} yaw drift")
        actor_metrics[actor_id] = {
            "checked_frame_count": frame_count,
            "maximum_position_error_cm": max_position,
            "maximum_yaw_error_deg": max_yaw,
        }
    return {
        "camera": {
            "checked_frame_count": frame_count,
            "unique_pose_hash_count": len(set(series.pose_hashes)),
            "maximum_position_error_cm": maximum_camera_position_error_cm,
            "maximum_yaw_error_deg": maximum_camera_yaw_error_deg,
        },
        "actors": actor_metrics,
    }


def _bind_native_pixel_capture(
    *,
    manifest_path: Path,
    facts: Mapping[str, Any],
    sensor_rig: Mapping[str, Any],
    audio_wav_path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest = _load_json(manifest_path)
    episode_id = facts["episode_id"]
    _require(
        manifest.get("schema") == "avengine_qa_native_spear_pixel_episode_v1"
        and manifest.get("status") == "pass"
        and manifest.get("native_pixel_fact_binding_claim") is True,
        "native pixel capture manifest did not pass",
    )
    _require(manifest.get("scenario_id") == episode_id, "pixel Episode identity drift")
    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, Mapping), "pixel manifest lacks artifacts")
    required_artifacts = {
        "native_rgb_binaural",
        "metric_depth",
        "pixel_masks",
        "pixel_visibility_truth",
        "runtime_readbacks",
        "normal_object_ids",
        "object_id_descriptors",
    }
    _require(
        required_artifacts <= set(artifacts),
        "pixel manifest lacks required retained artifacts",
    )
    paths = {name: Path(artifacts[name]).resolve() for name in required_artifacts}
    for name, path in paths.items():
        _require(path.is_file(), f"pixel artifact is missing: {name}={path}")

    artifact_records = manifest.get("artifact_records")
    _require(
        isinstance(artifact_records, Mapping)
        and set(artifact_records) == set(artifacts),
        "pixel manifest artifact records are incomplete",
    )
    hashes = manifest.get("sha256")
    _require(isinstance(hashes, Mapping), "pixel manifest lacks hashes")
    observed_file_hashes: dict[str, str] = {}
    for name, raw_path in artifacts.items():
        path = Path(raw_path).resolve()
        record = artifact_records[name]
        _require(
            isinstance(record, Mapping) and record.get("path") == str(path),
            f"pixel artifact record path drift: {name}",
        )
        if path.is_file():
            observed_hash = _sha256_file(path)
            observed_file_hashes[name] = observed_hash
            _require(
                record.get("kind") == "file"
                and record.get("size_bytes") == path.stat().st_size
                and record.get("sha256") == observed_hash,
                f"pixel artifact file record drift: {name}",
            )
        else:
            _require(path.is_dir(), f"pixel artifact is missing: {name}={path}")
            inventory = _directory_inventory(path)
            _require(
                record.get("kind") == "directory"
                and record.get("file_count") == len(inventory)
                and record.get("total_size_bytes")
                == sum(item["size_bytes"] for item in inventory)
                and record.get("inventory") == inventory
                and record.get("inventory_root_sha256")
                == canonical_json_sha256(inventory),
                f"pixel artifact directory inventory drift: {name}",
            )
    _require(
        dict(hashes) == observed_file_hashes,
        "pixel manifest file-hash inventory is incomplete",
    )
    for name in required_artifacts:
        _require(
            hashes.get(name) == _sha256_file(paths[name]),
            f"pixel artifact hash drift: {name}",
        )

    truth = _load_json(paths["pixel_visibility_truth"])
    fact_truth = facts.get("visibility", {}).get("pixel_truth")
    _require(
        isinstance(fact_truth, Mapping)
        and canonical_json_sha256(fact_truth) == canonical_json_sha256(truth),
        "Facts do not embed the exact native pixel truth",
    )
    authority = "same_renderer_same_camera_normal_vs_target_only_metric_depth_v1"
    _require(truth.get("authority") == authority, "pixel truth authority drift")
    _require(
        manifest.get("pixel_visibility", {}).get("authority") == authority,
        "pixel manifest authority drift",
    )
    frame_count = facts["time"]["frame_count"]
    resolution = tuple(truth["resolution_hw"])
    _require(
        resolution == tuple(facts["visibility"]["resolution_hw"]),
        "pixel resolution differs from Facts",
    )
    series = m7_sensor_rig_pose_series(sensor_rig)
    _require(
        truth["frame_indices"] == list(range(frame_count))
        and truth["camera_pose_ids"] == list(series.pose_hashes),
        "pixel frame or camera pose hashes differ from SensorRigTrajectory",
    )

    readbacks = _load_json(paths["runtime_readbacks"])
    normal_readbacks = readbacks.get("normal")
    target_readbacks = readbacks.get("target_only")
    _require(
        isinstance(normal_readbacks, list) and len(normal_readbacks) == frame_count,
        "pixel normal-pass runtime readbacks are incomplete",
    )
    _require(
        isinstance(target_readbacks, Mapping)
        and set(target_readbacks) == set(facts["tracks"]["instances"]),
        "pixel target-pass runtime readbacks have wrong instances",
    )
    for frame_index, record in enumerate(normal_readbacks):
        _require(
            record["camera"].get("expected_pose_hash") == series.pose_hashes[frame_index],
            f"pixel normal camera pose hash drift at frame {frame_index}",
        )
    for instance_id, records in target_readbacks.items():
        _require(
            isinstance(records, list) and len(records) == frame_count,
            f"{instance_id} pixel target readbacks are incomplete",
        )
        for frame_index, record in enumerate(records):
            _require(
                record["camera"].get("expected_pose_hash")
                == series.pose_hashes[frame_index],
                f"{instance_id} target camera pose hash drift at frame {frame_index}",
            )

    depth_contract = truth.get("depth_comparison", {})
    background = float(depth_contract.get("target_only_background_depth_m"))
    absolute_tolerance = float(depth_contract.get("absolute_tolerance_m"))
    relative_tolerance = float(depth_contract.get("relative_tolerance"))
    with np.load(paths["metric_depth"]) as depth_payload:
        required_depth_keys = {
            "normal_depth_m",
            "target_only_source1_depth_m",
            "target_only_source2_depth_m",
        }
        _require(
            required_depth_keys == set(depth_payload.files),
            "metric depth artifact must contain exactly the contracted arrays",
        )
        normal_depth = np.asarray(depth_payload["normal_depth_m"], dtype=np.float32)
        target_depths = {
            "source1": np.asarray(
                depth_payload["target_only_source1_depth_m"], dtype=np.float32
            ),
            "source2": np.asarray(
                depth_payload["target_only_source2_depth_m"], dtype=np.float32
            ),
        }
    expected_shape = (frame_count, *resolution)
    _require(
        normal_depth.shape == expected_shape
        and np.all(np.isfinite(normal_depth))
        and np.all(normal_depth > 0.0),
        "normal metric depth shape or finite-value gate failed",
    )
    for instance_id, target_depth in target_depths.items():
        _require(
            target_depth.shape == expected_shape
            and np.all(np.isfinite(target_depth))
            and np.all(target_depth > 0.0),
            f"{instance_id} target metric depth shape or finite-value gate failed",
        )

    modal_semantic = np.zeros(expected_shape, dtype=np.uint8)
    best_residual = np.full(expected_shape, np.inf, dtype=np.float32)
    target_footprints: dict[str, np.ndarray] = {}
    semantic_ids = {
        instance_id: int(entry["semantic_id"])
        for instance_id, entry in truth["per_instance"].items()
    }
    for instance_id in sorted(target_depths):
        target_depth = target_depths[instance_id]
        footprint = target_depth < background
        residual = np.abs(normal_depth - target_depth)
        visible = footprint & (
            residual <= absolute_tolerance + relative_tolerance * target_depth
        )
        wins = visible & (residual < best_residual)
        modal_semantic[wins] = semantic_ids[instance_id]
        best_residual[wins] = residual[wins]
        target_footprints[instance_id] = footprint

    with np.load(paths["pixel_masks"]) as mask_payload:
        required_mask_keys = {
            "depth_derived_modal_semantic",
            "modal_visible_source1",
            "modal_visible_source2",
            "target_only_source1",
            "target_only_source2",
        }
        _require(
            required_mask_keys == set(mask_payload.files),
            "pixel mask artifact must contain exactly the contracted arrays",
        )
        _require(
            np.array_equal(
                mask_payload["depth_derived_modal_semantic"], modal_semantic
            ),
            "retained modal semantic mask differs from metric-depth derivation",
        )
        for instance_id in sorted(target_depths):
            semantic_id = semantic_ids[instance_id]
            modal_visible = np.asarray(
                mask_payload[f"modal_visible_{instance_id}"], dtype=bool
            )
            target_mask = np.asarray(
                mask_payload[f"target_only_{instance_id}"]
            ) == semantic_id
            expected_visible = modal_semantic == semantic_id
            _require(
                modal_visible.shape == expected_shape
                and target_mask.shape == expected_shape
                and np.array_equal(modal_visible, expected_visible)
                and np.array_equal(target_mask, target_footprints[instance_id])
                and not np.any(modal_visible & ~target_mask),
                f"{instance_id} retained pixel masks fail depth/subset binding",
            )
            frames = truth["per_instance"][instance_id]["frames"]
            for frame_index, frame in enumerate(frames):
                visible_pixels = int(np.count_nonzero(modal_visible[frame_index]))
                target_pixels = int(np.count_nonzero(target_mask[frame_index]))
                _require(
                    visible_pixels == frame["visible_pixels"]
                    and target_pixels == frame["target_pixels"]
                    and 0 <= visible_pixels <= target_pixels,
                    f"{instance_id} pixel counts drift at frame {frame_index}",
                )

    _require(
        manifest.get("audio", {}).get("sha256") == _sha256_file(audio_wav_path),
        "pixel capture audio identity drift",
    )
    media_probe = _probe_media(paths["native_rgb_binaural"])
    _require(
        media_probe["video_codec"] == "h264"
        and media_probe["frame_count"] == frame_count
        and (media_probe["width"], media_probe["height"])
        == (resolution[1], resolution[0])
        and media_probe["audio_channels"] == facts["audio"]["channel_count"]
        and media_probe["audio_sample_rate_hz"]
        == facts["audio"]["sample_rate_hz"],
        "native pixel AV stream contract differs from Facts",
    )
    return (
        {
            "status": "computed_native_metric_depth",
            "authority": authority,
            "camera_pose_hash_count": len(series.pose_hashes),
            "runtime_alignment": manifest.get("runtime_alignment"),
            "state_counts": {
                instance_id: entry["state_counts"]
                for instance_id, entry in truth["per_instance"].items()
            },
            "media_readback": media_probe,
        },
        paths,
    )


def _make_question_inputs(facts: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    events = facts["sound_events"]
    _require(len(events) == 2, "native QA canary requires exactly two sound events")
    sound_ids: list[str] = []
    sound_assets: list[dict[str, Any]] = []
    bindings: dict[str, Any] = {}
    labels = {
        "dog": "clothov2_growling_barking_dog_v1",
        "cat": "clothov2_cat_meowing_v1",
    }
    for event in events:
        species = event["sound_class"]["species_id"]
        sound_id = labels.get(species, f"controlled_{species}_sound_v1")
        _require(sound_id not in sound_ids, "sound IDs are not unique")
        sound_ids.append(sound_id)
        dry = event["dry_variant"]
        sound_assets.append(
            {
                "sound_asset_id": sound_id,
                "species": species,
                "path": dry["input_path"],
                "sha256": dry["input_sha256"],
                "admissibility": "research",
            }
        )
        bindings[event["event_id"]] = {"sound_asset_id": sound_id}

    instances = {item["instance_id"]: item for item in facts["instances"]}
    source1 = instances["source1"]
    azimuths = facts["tracks"]["instances"]["source1"]["doa"]["azimuth_deg"]
    frame_index = max(range(len(azimuths)), key=lambda index: abs(float(azimuths[index])))
    specs = [
        {
            "schema": "avengine_qa_question_spec_v1",
            "spec_id": "QS-001",
            "question_type": "appearance_to_speaking",
            "selectors": {
                "appearance_field": "breed_id",
                "appearance_value": source1["breed_id"],
            },
        },
        {
            "schema": "avengine_qa_question_spec_v1",
            "spec_id": "QS-002",
            "question_type": "sound_to_appearance",
            "selectors": {
                "sound_asset_id": sound_ids[0],
                "appearance_field": "coat_value",
            },
        },
        {
            "schema": "avengine_qa_question_spec_v1",
            "spec_id": "QS-003",
            "question_type": "who_spoke_first",
            "selectors": {},
        },
        {
            "schema": "avengine_qa_question_spec_v1",
            "spec_id": "QS-004",
            "question_type": "speaker_side",
            "selectors": {"sound_asset_id": sound_ids[0], "frame_index": frame_index},
        },
        {
            "schema": "avengine_qa_question_spec_v1",
            "spec_id": "QS-005",
            "question_type": "overlapping_speech",
            "selectors": {"sound_asset_ids": sound_ids},
        },
        {
            "schema": "avengine_qa_question_spec_v1",
            "spec_id": "QS-006",
            "question_type": "speaking_while_moving",
            "selectors": {"sound_asset_id": sound_ids[1]},
        },
        {
            "schema": "avengine_qa_question_spec_v1",
            "spec_id": "QS-007",
            "question_type": "offscreen_to_onscreen",
            "selectors": {"target_instance_id": "source1"},
        },
        {
            "schema": "avengine_qa_question_spec_v1",
            "spec_id": "QS-008",
            "question_type": "occlusion_while_speaking",
            "selectors": {"sound_asset_id": sound_ids[0], "frame_index": frame_index},
        },
        {
            "schema": "avengine_qa_question_spec_v1",
            "spec_id": "QS-009",
            "question_type": "reappeared_after_occlusion",
            "selectors": {"target_instance_id": "source1"},
        },
        {
            "schema": "avengine_qa_question_spec_v1",
            "spec_id": "QS-010",
            "question_type": "occluder_identity",
            "selectors": {"target_instance_id": "source1", "frame_index": frame_index},
        },
    ]
    registry = {
        "schema": "avengine_qa_controlled_sound_registry_v1",
        "registry_id": "lead_a_native_dynamic_episode_sounds_v1",
        "sound_assets": sound_assets,
    }
    return specs, registry, bindings


def build(
    *,
    facts_path: Path,
    facts_stats_path: Path,
    sensor_rig_path: Path,
    plan_delivery_path: Path,
    audio_delivery_path: Path,
    audio_wav_path: Path,
    bundle_manifest_path: Path,
    habitat_clean_video_path: Path,
    ue_evidence_path: Path,
    runtime_readbacks_path: Path,
    ue_video_path: Path,
    asset_registry_path: Path,
    output: Path,
    native_pixel_manifest_path: Path | None = None,
) -> dict[str, Any]:
    inputs = {
        "facts": facts_path,
        "facts_stats": facts_stats_path,
        "sensor_rig": sensor_rig_path,
        "plan_delivery": plan_delivery_path,
        "audio_delivery": audio_delivery_path,
        "audio_wav": audio_wav_path,
        "bundle_manifest": bundle_manifest_path,
        "habitat_clean_video": habitat_clean_video_path,
        "ue_evidence": ue_evidence_path,
        "runtime_readbacks": runtime_readbacks_path,
        "ue_video": ue_video_path,
        "asset_registry": asset_registry_path,
    }
    for path in inputs.values():
        _require(path.is_file(), f"input is missing: {path}")

    facts = _load_json(facts_path)
    facts_stats = _load_json(facts_stats_path)
    sensor_rig = _load_json(sensor_rig_path)
    plan_delivery = _load_json(plan_delivery_path)
    audio_delivery = _load_json(audio_delivery_path)
    bundle_manifest = _load_json(bundle_manifest_path)
    ue_evidence = _load_json(ue_evidence_path)
    runtime_readbacks = _load_json(runtime_readbacks_path)
    asset_registry = _load_json(asset_registry_path)
    episode_id = facts.get("episode_id")
    _require(
        facts.get("schema") == "avengine_qa_fact_table_v1"
        and facts.get("status") == "pass",
        "Facts are not a passing QA Fact table",
    )
    _require(facts["listener"].get("static") is False, "Listener is not dynamic")
    binding = m7_sensor_rig_binding(sensor_rig)
    fact_binding = facts["listener"]["sensor_rig_trajectory"]
    for field in ("trajectory_id", "first_pose_hash", "last_pose_hash", "pose_hash_algorithm"):
        _require(fact_binding.get(field) == binding.get(field), f"Facts {field} drift")
    _require(
        facts_stats.get("plan_listener_pose_checks")
        == plan_delivery.get("requested_pair_state_count")
        == audio_delivery.get("sensor_rig_rir_alignment", {}).get("checked_use_count"),
        "RIR Listener checks do not close over every planned pair state",
    )
    _require(
        plan_delivery.get("sensor_rig_trajectory", {}).get("content_sha256")
        == binding["content_sha256"],
        "plan SensorRig content identity drift",
    )
    _require(
        audio_delivery.get("sensor_rig_trajectory", {}).get("content_sha256")
        == binding["content_sha256"],
        "audio SensorRig content identity drift",
    )
    _require(
        bundle_manifest.get("sensor_rig_trajectory", {}).get("content_sha256")
        == binding["content_sha256"],
        "visual bundle SensorRig content identity drift",
    )
    _require(
        bundle_manifest.get("episode_ids") == [episode_id],
        "visual bundle Episode identity drift",
    )
    audio_wav_sha = _sha256_file(audio_wav_path)
    _require(audio_wav_sha == facts["audio"]["mixture_sha256"], "Facts audio hash drift")

    _require(ue_evidence.get("status") == "pass", "native UE scenario did not pass")
    _require(ue_evidence.get("scenario_id") == episode_id, "UE Episode identity drift")
    root_gate = ue_evidence.get("root_readback", {})
    _require(root_gate.get("camera", {}).get("status") == "pass", "UE camera gate failed")
    for actor_id in ("source1_actor", "source2_actor"):
        _require(root_gate.get(actor_id, {}).get("status") == "pass", f"{actor_id} root gate failed")
        _require(
            ue_evidence.get("animation_phase_readback", {}).get(actor_id, {}).get("status") == "pass",
            f"{actor_id} animation gate failed",
        )
    alignment = _check_runtime_alignment(
        facts=facts,
        sensor_rig=sensor_rig,
        runtime_readbacks=runtime_readbacks,
    )

    media_record = ue_evidence.get("media", {}).get("ue_clean_binaural", {})
    _require(media_record.get("status") == "pass", "native AV media gate failed")
    _require(
        media_record.get("size_bytes") == ue_video_path.stat().st_size,
        "native AV media size drift",
    )
    media_probe = _probe_media(ue_video_path)
    _require(media_probe["video_codec"] == "h264", "native RGB stream is not H.264")
    _require(
        media_probe["frame_count"] == facts["time"]["frame_count"]
        and media_probe["frame_rate"] == f"{facts['time']['frame_rate_hz']}/1"
        and media_probe["audio_channels"] == facts["audio"]["channel_count"]
        and media_probe["audio_sample_rate_hz"] == facts["audio"]["sample_rate_hz"],
        "native AV stream contract differs from Facts",
    )
    source_packet_sha = _audio_packet_sha256(habitat_clean_video_path)
    _require(
        media_probe["audio_packet_sha256"]
        == source_packet_sha
        == media_record.get("audio_packet_sha256"),
        "native UE video did not preserve the authoritative binaural packets",
    )

    native_pixel_binding: dict[str, Any] | None = None
    native_pixel_inputs: dict[str, Path] = {}
    if native_pixel_manifest_path is not None:
        _require(
            native_pixel_manifest_path.is_file(),
            f"native pixel manifest is missing: {native_pixel_manifest_path}",
        )
        native_pixel_binding, native_pixel_inputs = _bind_native_pixel_capture(
            manifest_path=native_pixel_manifest_path,
            facts=facts,
            sensor_rig=sensor_rig,
            audio_wav_path=audio_wav_path,
        )
        inputs["native_pixel_manifest"] = native_pixel_manifest_path
        inputs.update(
            {
                f"native_pixel_{name}": path
                for name, path in native_pixel_inputs.items()
            }
        )

    specs, sound_registry, event_bindings = _make_question_inputs(facts)
    evaluations = evaluate_question_specs(
        specs,
        facts=facts,
        asset_registry=asset_registry,
        sound_registry=sound_registry,
        event_sound_bindings=event_bindings,
    )
    actual_status = {item["spec_id"]: item["status"] for item in evaluations}
    expected_status = (
        {
            "QS-001": "pass",
            "QS-002": "pass",
            "QS-003": "rejected",
            "QS-004": "pass",
            "QS-005": "pass",
            "QS-006": "pass",
            "QS-007": "pass",
            "QS-008": "pass",
            "QS-009": "pass",
            "QS-010": "unsupported",
        }
        if native_pixel_binding is not None
        else {
            "QS-001": "pass",
            "QS-002": "pass",
            "QS-003": "rejected",
            "QS-004": "pass",
            "QS-005": "pass",
            "QS-006": "pass",
            "QS-007": "rejected",
            "QS-008": "rejected",
            "QS-009": "rejected",
            "QS-010": "rejected",
        }
    )
    _require(actual_status == expected_status, f"QuestionSpec status drift: {actual_status}")

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "question_specs.json", specs)
    write_json(output / "controlled_sound_registry.json", sound_registry)
    write_json(output / "event_sound_bindings.json", event_bindings)
    write_json(output / "expected_question_status.json", expected_status)
    write_json(output / "question_evaluations.json", evaluations)
    manifest = {
        "schema": SCHEMA,
        "status": "pass",
        "qualification_claim": False,
        "claim_boundary": (
            "Retained research canary binding: real native SPEAR/UE RGB and copied "
            "Habitat/RLR binaural audio are frame-bound to QA Facts; no dataset admission"
        ),
        "episode_id": episode_id,
        "backend_authority": {
            "rgb": "native_spear_unreal",
            "audio": "habitat_native_rlr_binaural_packet_preserved_in_ue_mux",
            "trajectory_and_labels": "habitat_native_sensor_rig_trajectory",
        },
        "sensor_rig": binding,
        "runtime_alignment": alignment,
        "media_readback": media_probe,
        "question_status_by_spec": actual_status,
        "pixel_visibility": (
            native_pixel_binding
            if native_pixel_binding is not None
            else {
                "status": "not_run",
                "reason": (
                    "the retained SPEAR/UE RGB run did not capture same-renderer "
                    "normal plus target-only evidence; hermetic or Habitat masks "
                    "cannot be attached to these UE pixels"
                ),
            }
        ),
        "inputs": {name: _file_record(path) for name, path in inputs.items()},
        "outputs": {
            "question_specs": "question_specs.json",
            "controlled_sound_registry": "controlled_sound_registry.json",
            "event_sound_bindings": "event_sound_bindings.json",
            "expected_question_status": "expected_question_status.json",
            "question_evaluations": "question_evaluations.json",
        },
        "producer": {
            "tool": "tools/qa/bind_native_spear_episode.py",
            "contract_sha256": canonical_json_sha256(
                {
                    "schema": SCHEMA,
                    "episode_id": episode_id,
                    "sensor_rig_content_sha256": binding["content_sha256"],
                    "facts_sha256": _sha256_file(facts_path),
                    "ue_video_sha256": _sha256_file(ue_video_path),
                    "native_pixel_manifest_sha256": (
                        None
                        if native_pixel_manifest_path is None
                        else _sha256_file(native_pixel_manifest_path)
                    ),
                }
            ),
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "facts",
        "facts_stats",
        "sensor_rig",
        "plan_delivery",
        "audio_delivery",
        "audio_wav",
        "bundle_manifest",
        "habitat_clean_video",
        "ue_evidence",
        "runtime_readbacks",
        "ue_video",
        "asset_registry",
        "output",
    ):
        parser.add_argument(f"--{option.replace('_', '-')}", required=True, type=Path)
    parser.add_argument("--native-pixel-manifest", type=Path)
    args = parser.parse_args()
    manifest = build(
        facts_path=args.facts.resolve(),
        facts_stats_path=args.facts_stats.resolve(),
        sensor_rig_path=args.sensor_rig.resolve(),
        plan_delivery_path=args.plan_delivery.resolve(),
        audio_delivery_path=args.audio_delivery.resolve(),
        audio_wav_path=args.audio_wav.resolve(),
        bundle_manifest_path=args.bundle_manifest.resolve(),
        habitat_clean_video_path=args.habitat_clean_video.resolve(),
        ue_evidence_path=args.ue_evidence.resolve(),
        runtime_readbacks_path=args.runtime_readbacks.resolve(),
        ue_video_path=args.ue_video.resolve(),
        asset_registry_path=args.asset_registry.resolve(),
        output=args.output.resolve(),
        native_pixel_manifest_path=(
            None
            if args.native_pixel_manifest is None
            else args.native_pixel_manifest.resolve()
        ),
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
