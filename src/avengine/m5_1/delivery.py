"""Evidence-bound helpers for the final M5.1 legacy review delivery."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256, load_json, sha256_file
from avengine.m4.audio import AudioContractError
from avengine.m5.acoustics import DynamicRIRSequence
from avengine.m5_1.acoustics import ResearchReviewKeyframeGrid
from avengine.m5_1.dry_audio import DryAudioClipSpec
from avengine.m5_1.review import SourceOverlayTrack


DELIVERY_SCHEMA = "avengine_m5_1_legacy_delivery_v1"
SOURCE_ANCHOR_INDEX = {"source0": 1, "source1": 2}
SOURCE_SEMANTIC_ID = {"source0": 220, "source1": 221}
SOURCE_LABEL = {"source0": "HUMAN", "source1": "BEAGLE"}
SOURCE_COLOR = {"source0": (42, 210, 220), "source1": (250, 120, 70)}
SOURCE_GATE = {
    "source0": "human_center_point_aabb",
    "source1": "dog_center_point_aabb",
}


class M51DeliveryError(ValueError):
    """Retained M5.1 delivery inputs or outputs disagree."""


def semantic_centroid_track(semantic_frames: Any, semantic_id: int) -> np.ndarray:
    """Return finite image centroids or NaN pairs for one semantic actor ID."""

    semantic = np.asarray(semantic_frames)
    if semantic.ndim != 3 or semantic.shape[0] < 1 or semantic.dtype.kind not in "iu":
        raise M51DeliveryError("semantic frames must be nonempty integer [frame,h,w]")
    if isinstance(semantic_id, bool) or not isinstance(semantic_id, int) or semantic_id < 0:
        raise M51DeliveryError("semantic_id must be a nonnegative integer")
    result = np.full((semantic.shape[0], 2), np.nan, dtype=np.float64)
    for frame_index, frame in enumerate(semantic):
        y, x = np.nonzero(frame == semantic_id)
        if x.size:
            result[frame_index] = (float(np.mean(x)), float(np.mean(y)))
    return np.ascontiguousarray(result)


def event_overlay_state(
    source: Mapping[str, Any], frame_count: int
) -> tuple[tuple[str | None, ...], tuple[bool, ...]]:
    """Expand half-open source event windows into one review state per frame."""

    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise M51DeliveryError("frame_count must be a positive integer")
    events = source.get("event_windows")
    if not isinstance(events, list):
        raise M51DeliveryError("source event_windows must be an array")
    state: list[str | None] = [None] * frame_count
    for event in events:
        if not isinstance(event, Mapping):
            raise M51DeliveryError("source event window must be an object")
        event_id = event.get("event_id")
        start = event.get("start_frame")
        end = event.get("end_frame_exclusive")
        if (
            not isinstance(event_id, str)
            or not event_id
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not 0 <= start < end <= frame_count
        ):
            raise M51DeliveryError("source event frame interval is invalid")
        for frame_index in range(start, end):
            if state[frame_index] is not None:
                raise M51DeliveryError(
                    "one source has overlapping event windows; review state is ambiguous"
                )
            state[frame_index] = event_id
    return tuple(state), tuple(value is not None for value in state)


def _present_flags(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        raise M51DeliveryError("flag collection must be an object")
    return tuple(
        sorted(
            (
                str(flag_id)
                for flag_id, assessment in value.items()
                if isinstance(assessment, Mapping)
                and assessment.get("status") == "present"
                and assessment.get("value") is True
            ),
            key=lambda item: item.encode("ascii"),
        )
    )


def build_legacy_overlay_tracks(
    source_manifest: Mapping[str, Any],
    route_manifest: Mapping[str, Any],
    *,
    anchor_positions_m: Any,
    semantic_frames: Any,
) -> tuple[SourceOverlayTrack, ...]:
    """Bind declared source/event metadata to captured link positions and pixels."""

    anchors = np.asarray(anchor_positions_m, dtype=np.float64)
    semantic = np.asarray(semantic_frames)
    frame_count = int(source_manifest.get("clip", {}).get("frame_count", 0))
    if (
        anchors.shape != (frame_count, 3, 3)
        or not np.all(np.isfinite(anchors))
        or semantic.ndim != 3
        or semantic.shape[0] != frame_count
    ):
        raise M51DeliveryError("capture anchors/semantic differ from source clip")
    sources = source_manifest.get("sources")
    if not isinstance(sources, list) or {
        item.get("source_id") for item in sources if isinstance(item, Mapping)
    } != set(SOURCE_ANCHOR_INDEX):
        raise M51DeliveryError("legacy delivery requires exactly source0 and source1")
    gates = route_manifest.get("gates")
    if not isinstance(gates, Mapping):
        raise M51DeliveryError("route manifest lacks center-point gates")
    tracks: list[SourceOverlayTrack] = []
    for source in sources:
        assert isinstance(source, Mapping)
        source_id = str(source["source_id"])
        event_ids, active = event_overlay_state(source, frame_count)
        asset_class = str(source.get("asset_class"))
        taxonomy = (
            source.get("voice_taxonomy")
            if asset_class == "human"
            else source.get("call_taxonomy")
        )
        if not isinstance(taxonomy, Mapping):
            raise M51DeliveryError(f"{source_id} taxonomy is missing")
        sound_class = str(
            taxonomy.get("vocalization_type", taxonomy.get("call_type", "unknown"))
        )
        gate = gates.get(SOURCE_GATE[source_id])
        frames = gate.get("frames") if isinstance(gate, Mapping) else None
        if not isinstance(frames, list) or len(frames) != frame_count:
            raise M51DeliveryError(f"{source_id} center-point gate differs from clip")
        clearance = np.asarray(
            [frame.get("minimum_clearance_m") for frame in frames],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(clearance)) or np.any(clearance < 0.0):
            raise M51DeliveryError(f"{source_id} center-point clearance is invalid")
        tracks.append(
            SourceOverlayTrack(
                source_id=source_id,
                label=SOURCE_LABEL[source_id],
                asset_class=asset_class,
                sound_class=sound_class,
                color_rgb=SOURCE_COLOR[source_id],
                positions_m=np.ascontiguousarray(
                    anchors[:, SOURCE_ANCHOR_INDEX[source_id], :]
                ),
                current_event_by_frame=event_ids,
                active_by_frame=active,
                true_flags=_present_flags(source.get("flags")),
                center_clearance_m=np.ascontiguousarray(clearance),
                main_marker_xy=semantic_centroid_track(
                    semantic, SOURCE_SEMANTIC_ID[source_id]
                ),
            )
        )
    return tuple(sorted(tracks, key=lambda track: track.source_id.encode("ascii")))


def declared_audio_asset_bindings(
    source_manifest: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Resolve authenticated local file URIs declared by the source contract."""

    result: dict[str, dict[str, str]] = {}
    sources = source_manifest.get("sources")
    if not isinstance(sources, list):
        raise M51DeliveryError("source manifest sources must be an array")
    for source in sources:
        provenance = source.get("provenance") if isinstance(source, Mapping) else None
        assets = provenance.get("audio_assets") if isinstance(provenance, Mapping) else None
        if not isinstance(assets, list):
            raise M51DeliveryError("source provenance audio_assets must be an array")
        for asset in assets:
            if not isinstance(asset, Mapping):
                raise M51DeliveryError("audio asset provenance must be an object")
            asset_id = asset.get("asset_id")
            uri = asset.get("uri")
            digest = asset.get("sha256")
            parsed = urlparse(uri) if isinstance(uri, str) else None
            if (
                not isinstance(asset_id, str)
                or not asset_id
                or parsed is None
                or parsed.scheme != "file"
                or parsed.netloc not in {"", "localhost"}
                or not isinstance(digest, str)
            ):
                raise M51DeliveryError("audio asset requires an authenticated local file URI")
            path = Path(unquote(parsed.path)).resolve()
            if not path.is_file() or sha256_file(path) != digest:
                raise M51DeliveryError(f"audio asset does not match provenance: {asset_id}")
            binding = {"path": str(path), "sha256": digest}
            if asset_id in result and result[asset_id] != binding:
                raise M51DeliveryError(f"audio asset ID is ambiguous: {asset_id}")
            result[asset_id] = binding
    return result


def executable_event_mappings(
    source_manifest: Mapping[str, Any],
    *,
    gain_by_source: Mapping[str, float],
    fade_samples: int,
) -> tuple[dict[str, Any], ...]:
    """Attach explicit source-native slices, gains and fades to declared events."""

    if isinstance(fade_samples, bool) or not isinstance(fade_samples, int) or fade_samples < 0:
        raise M51DeliveryError("fade_samples must be a nonnegative integer")
    sources = source_manifest.get("sources")
    if not isinstance(sources, list):
        raise M51DeliveryError("source manifest sources must be an array")
    observed = {item.get("source_id") for item in sources if isinstance(item, Mapping)}
    if set(gain_by_source) != observed:
        raise M51DeliveryError("gain_by_source must cover every and only declared source")
    events: list[dict[str, Any]] = []
    for source in sources:
        assert isinstance(source, Mapping)
        source_id = str(source["source_id"])
        gain = gain_by_source[source_id]
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(
            float(gain)
        ) or float(gain) < 0.0:
            raise M51DeliveryError(f"{source_id} gain must be finite and nonnegative")
        for event in source.get("event_windows", []):
            if not isinstance(event, Mapping):
                raise M51DeliveryError("event window must be an object")
            program = event.get("audio_program")
            if not isinstance(program, Mapping):
                raise M51DeliveryError("event lacks an explicit audio_program")
            executable = deepcopy(dict(event))
            executable["dry_clip_start_sample"] = program.get("source_start_sample")
            executable["dry_clip_end_sample_exclusive"] = program.get(
                "source_end_sample_exclusive"
            )
            executable["linear_gain"] = float(gain)
            executable["fade_samples"] = fade_samples
            events.append(executable)
    return tuple(events)


def verify_audio_program_receipts(
    source_manifest: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> None:
    """Verify executable resampling/crop/pad facts against the source declaration."""

    declared: dict[str, Mapping[str, Any]] = {}
    for source in source_manifest.get("sources", []):
        for event in source.get("event_windows", []):
            declared[str(event["event_id"])] = event["audio_program"]
    if {str(item.get("event_id")) for item in receipts} != set(declared):
        raise M51DeliveryError("dry assembly receipt event IDs differ from source manifest")
    for receipt in receipts:
        event_id = str(receipt["event_id"])
        program = declared[event_id]
        observed = receipt["resampling"]
        fit = receipt["fit"]
        expected = {
            "source_sample_rate_hz": int(program["source_sample_rate_hz"]),
            "target_sample_rate_hz": int(program["render_sample_rate_hz"]),
            "output_sample_count": int(program["resampled_content_sample_count"]),
            "zero_padded_tail_sample_count": int(program["tail_padding_samples"]),
        }
        actual = {
            "source_sample_rate_hz": int(observed["source_sample_rate_hz"]),
            "target_sample_rate_hz": int(observed["target_sample_rate_hz"]),
            "output_sample_count": int(observed["output_sample_count"]),
            "zero_padded_tail_sample_count": int(fit["zero_padded_tail_sample_count"]),
        }
        if actual != expected or int(fit["cropped_tail_sample_count"]) != 0:
            raise M51DeliveryError(
                f"event {event_id} dry assembly differs from declared audio_program"
            )


def load_retained_binaural_sequence(
    acoustics_dir: str | Path,
    *,
    grid: ResearchReviewKeyframeGrid,
) -> DynamicRIRSequence:
    """Authenticate and reconstruct one retained M5.1 binaural RIR sequence."""

    root = Path(acoustics_dir).resolve()
    evidence = load_json(root / "evidence.json")
    if evidence.get("status") != "pass" or evidence.get("qualification_claim") is not False:
        raise M51DeliveryError("retained RIR evidence is not a bounded pass")
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise M51DeliveryError("retained RIR evidence lacks artifacts")

    def artifact_path(role: str) -> Path:
        record = artifacts.get(role)
        path_value = record.get("path") if isinstance(record, Mapping) else None
        if not isinstance(path_value, str):
            raise M51DeliveryError(f"retained RIR artifact {role} is missing")
        path = (root / path_value).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise M51DeliveryError("retained RIR artifact escapes its bundle") from exc
        if (
            not path.is_file()
            or path.stat().st_size != record.get("byte_size")
            or sha256_file(path) != record.get("sha256")
        ):
            raise M51DeliveryError(f"retained RIR artifact {role} changed")
        return path

    samples = np.load(artifact_path("rir_samples"), allow_pickle=False)
    lengths = np.load(artifact_path("rir_lengths"), allow_pickle=False)
    metadata = load_json(artifact_path("rir_metadata"))
    trajectory = metadata.get("trajectory")
    keyframes = trajectory.get("keyframes") if isinstance(trajectory, Mapping) else None
    if not isinstance(keyframes, list):
        raise M51DeliveryError("retained RIR metadata lacks keyframes")
    sequence = DynamicRIRSequence(
        samples=np.ascontiguousarray(samples),
        lengths=np.ascontiguousarray(lengths),
        source_ids=tuple(metadata.get("source_ids", ())),
        keyframe_ticks=tuple(int(item["tick"]) for item in keyframes),
        keyframe_samples=tuple(int(item["sample_index"]) for item in keyframes),
        sample_rate_hz=int(metadata.get("sample_rate_hz", 0)),
        layout_type=str(metadata.get("layout_type")),
        layout_id=str(metadata.get("layout_id")),
        channel_labels=tuple(metadata.get("channel_labels", ())),
        trajectory_sha256=str(metadata.get("trajectory_sha256")),
        metadata=metadata,
    )
    if (
        sequence.source_ids != grid.source_ids
        or sequence.keyframe_ticks != tuple(frame.tick for frame in grid.keyframes)
        or sequence.keyframe_samples
        != tuple(frame.sample_index for frame in grid.keyframes)
        or sequence.sample_rate_hz != grid.sample_rate_hz
        or evidence.get("trajectory_sha256") != sequence.trajectory_sha256
    ):
        raise M51DeliveryError("retained RIR sequence differs from captured trajectory grid")
    return sequence


def binaural_frame_diagnostics(
    mixture: Any,
    clip: DryAudioClipSpec,
    *,
    maximum_itd_samples: int = 16,
) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
    """Measure frame-local ILD and broadband cross-correlation ITD for review."""

    audio = np.asarray(mixture, dtype=np.float64)
    if audio.shape != (2, clip.sample_count) or not np.all(np.isfinite(audio)):
        raise AudioContractError("binaural mixture must be finite [2,clip_samples]")
    if (
        isinstance(maximum_itd_samples, bool)
        or not isinstance(maximum_itd_samples, int)
        or maximum_itd_samples < 0
    ):
        raise AudioContractError("maximum_itd_samples must be nonnegative")
    labels: list[str] = []
    records: list[dict[str, Any]] = []
    epsilon = 1.0e-12
    for frame_index in range(clip.frame_count):
        start = clip.sample_boundary(frame_index)
        end = clip.sample_boundary(frame_index + 1)
        left = audio[0, start:end]
        right = audio[1, start:end]
        peak = float(max(np.max(np.abs(left)), np.max(np.abs(right))))
        if peak <= epsilon:
            labels.append("silent")
            records.append(
                {
                    "frame_index": frame_index,
                    "start_sample": start,
                    "end_sample_exclusive": end,
                    "active": False,
                    "peak_absolute": peak,
                    "ild_db": None,
                    "itd_xcorr_samples": None,
                    "itd_xcorr_us": None,
                    "xcorr_coefficient": None,
                }
            )
            continue
        left_rms = math.sqrt(float(np.mean(left * left)))
        right_rms = math.sqrt(float(np.mean(right * right)))
        ild = 20.0 * math.log10((left_rms + epsilon) / (right_rms + epsilon))
        best_lag = 0
        best_correlation = -math.inf
        for lag in range(-maximum_itd_samples, maximum_itd_samples + 1):
            if lag < 0:
                first, second = left[:lag], right[-lag:]
            elif lag > 0:
                first, second = left[lag:], right[:-lag]
            else:
                first, second = left, right
            norm = math.sqrt(float(np.dot(first, first) * np.dot(second, second)))
            correlation = float(np.dot(first, second) / norm) if norm > epsilon else -1.0
            if correlation > best_correlation + 1.0e-15 or (
                math.isclose(
                    correlation,
                    best_correlation,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
                and abs(lag) < abs(best_lag)
            ):
                best_lag = lag
                best_correlation = correlation
        itd_us = best_lag * 1_000_000.0 / clip.sample_rate_hz
        labels.append(
            f"ILD={ild:+.2f}dB ITD_xcorr={itd_us:+.1f}us peak={peak:.3f}"
        )
        records.append(
            {
                "frame_index": frame_index,
                "start_sample": start,
                "end_sample_exclusive": end,
                "active": True,
                "peak_absolute": peak,
                "ild_db": ild,
                "itd_xcorr_samples": best_lag,
                "itd_xcorr_us": itd_us,
                "xcorr_coefficient": best_correlation,
            }
        )
    return tuple(labels), tuple(records)


def actual_emitter_trajectory_record(
    anchor_positions_m: Any,
    *,
    capture_evidence_sha256: str,
) -> dict[str, Any]:
    """Return a compact hash-bound mapping from source IDs to animated links."""

    anchors = np.asarray(anchor_positions_m, dtype=np.float64)
    if anchors.ndim != 3 or anchors.shape[1:] != (3, 3) or not np.all(np.isfinite(anchors)):
        raise M51DeliveryError("actual emitter anchors must be finite [frame,3,3]")
    links = {"source0": "Bip01 MJaw", "source1": "beagle Xtra Mouth"}
    sources: dict[str, Any] = {}
    for source_id in sorted(SOURCE_ANCHOR_INDEX):
        positions = anchors[:, SOURCE_ANCHOR_INDEX[source_id], :]
        content = {
            "source_id": source_id,
            "link_name": links[source_id],
            "position_authority": "animated_articulated_link_world_transform_readback",
            "coordinate_frame": "avengine_world_right_handed_y_up_m",
            "frame_count": int(anchors.shape[0]),
            "positions_m": positions.tolist(),
        }
        content["trajectory_content_sha256"] = canonical_json_sha256(content)
        sources[source_id] = content
    record: dict[str, Any] = {
        "schema": "avengine_m5_1_actual_emitter_trajectories_v1",
        "capture_evidence_sha256": capture_evidence_sha256,
        "source_ids": sorted(sources),
        "sources": sources,
    }
    record["record_content_sha256"] = canonical_json_sha256(record)
    return record


__all__ = [
    "DELIVERY_SCHEMA",
    "M51DeliveryError",
    "SOURCE_ANCHOR_INDEX",
    "actual_emitter_trajectory_record",
    "binaural_frame_diagnostics",
    "build_legacy_overlay_tracks",
    "declared_audio_asset_bindings",
    "event_overlay_state",
    "executable_event_mappings",
    "load_retained_binaural_sequence",
    "semantic_centroid_track",
    "verify_audio_program_receipts",
]
