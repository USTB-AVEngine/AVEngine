#!/usr/bin/env python3
"""Validate and finalize representative runtime evidence for a room pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

from avengine.contracts.json_io import sha256_file


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def _visual_receipt(root, expected_timeline=None):
    root = Path(root).resolve()
    receipt = _read(root / "research_receipt.json")
    if receipt.get("status") != "research_only":
        raise RuntimeError(f"visual receipt is not research_only: {root}")
    capture = receipt.get("capture") or {}
    frame_count = capture.get("completed_frame_count")
    declared_count = capture.get("frame_count")
    if (isinstance(frame_count, bool) or not isinstance(frame_count, int)
            or frame_count < 1 or declared_count != frame_count):
        raise RuntimeError(f"visual receipt has no consistent frame count: {root}")
    if expected_timeline is not None:
        timeline = _read(Path(expected_timeline))
        expected_render = timeline.get("render") or {}
        expected_count = expected_render.get("frame_count")
        if expected_count != frame_count:
            raise RuntimeError(
                f"visual receipt closes {frame_count} frames but timeline declares "
                f"{expected_count}: {root}")
    return {
        "root": str(root),
        "receipt": str((root / "research_receipt.json").resolve()),
        "frame_records": str((root / "frame_records.json").resolve()),
        "frame_count": frame_count,
        "frame_rate_hz": capture.get("frame_rate_hz"),
    }


def _audio_receipt(root):
    root = Path(root).resolve()
    receipt_path = root / "research_receipt.json"
    receipt = _read(receipt_path)
    mixture = root / "audio/binaural/mixture.wav"
    if receipt.get("status") != "pass" or not receipt.get("research_only"):
        raise RuntimeError(f"audio receipt is not a research pass: {root}")
    if not mixture.is_file():
        raise RuntimeError(f"audio mixture is missing: {mixture}")
    audio = receipt.get("audio") or {}
    sample_rate = audio.get("sample_rate_hz")
    sample_count = audio.get("sample_count")
    if (isinstance(sample_rate, bool) or not isinstance(sample_rate, int)
            or sample_rate < 1 or isinstance(sample_count, bool)
            or not isinstance(sample_count, int) or sample_count < 1):
        raise RuntimeError(f"audio receipt has no valid sample clock: {root}")
    try:
        info = sf.info(mixture)
    except RuntimeError as exc:
        raise RuntimeError(f"cannot inspect audio mixture: {mixture}") from exc
    if (info.samplerate != sample_rate or info.frames != sample_count
            or info.channels != 2):
        raise RuntimeError(
            f"audio media clock differs from receipt at {root}: "
            f"media={info.samplerate}Hz/{info.frames} samples/{info.channels}ch, "
            f"receipt={sample_rate}Hz/{sample_count} samples/2ch")
    return {
        "root": str(root),
        "receipt": str(receipt_path.resolve()),
        "mixture": str(mixture.resolve()),
        "mixture_sha256": sha256_file(mixture),
        "event_count": receipt["audio_program"]["event_count"],
        "keyframe_count": receipt["rir"]["keyframe_count"],
        "sample_rate_hz": sample_rate,
        "sample_count": sample_count,
        "duration_seconds": sample_count / sample_rate,
    }


def _candidate_index(pilot):
    result = {}
    for room in pilot["rooms"].values():
        for profile in room["profiles"].values():
            for candidate in profile.get("candidates", []):
                result[candidate["pilot_id"]] = candidate
    return result


def _program_checks(candidate):
    main = _read(candidate["artifacts"]["main_program"])
    gatea = _read(candidate["artifacts"]["gatea_program"])
    main_times = [event["start_sample"] for event in main["events"]]
    gatea_times = [event["start_sample"] for event in gatea["events"]]
    main_sounds = sorted(event["sound_asset_id"] for event in main["events"])
    gatea_sounds = sorted(event["sound_asset_id"] for event in gatea["events"])
    assignments_changed = [
        (event["source_endpoint_id"], event["sound_asset_id"])
        for event in main["events"]
    ] != [
        (event["source_endpoint_id"], event["sound_asset_id"])
        for event in gatea["events"]
    ]
    checks = {
        "event_count_preserved": len(main["events"]) == len(gatea["events"]),
        "event_times_preserved": main_times == gatea_times,
        "sound_asset_multiset_preserved": main_sounds == gatea_sounds,
        "assignment_changed": assignments_changed,
    }
    if not all(checks.values()):
        raise RuntimeError(f"main/Gate-A program checks failed: {checks}")
    return checks


def _pixel_join(path, expected_fact):
    path = Path(path).resolve()
    value = _read(path)
    if value.get("status") != "pass":
        raise RuntimeError(f"selected pixel join is not pass: {path}")
    actual = Path(value["inputs"]["fact"]["path"]).resolve()
    if actual != Path(expected_fact).resolve():
        raise RuntimeError(
            f"pixel join fact differs from selected candidate: {actual}")
    return {
        "path": str(path),
        "status": value["status"],
        "bindings": value["bindings"],
        "inputs": value["inputs"],
    }


def _card17_distinct(first, second):
    first_frames = _read(Path(first) / "frame_records.json")["frames"]
    second_frames = _read(Path(second) / "frame_records.json")["frames"]
    def signature(frames):
        return [
            (
                frame["camera_pose"],
                {
                    slot: tuple(record["location_cm"])
                    for slot, record in frame["actor_anchor_poses"].items()
                },
            )
            for frame in frames
        ]
    if signature(first_frames) == signature(second_frames):
        raise RuntimeError("card17 segment runtime readbacks are identical")
    return {
        "runtime_readbacks_differ": True,
        "segment1_camera": first_frames[0]["camera_pose"],
        "segment2_camera": second_frames[0]["camera_pose"],
    }


def finalize(pilot, spec):
    candidates = _candidate_index(pilot)
    results = {}
    for profile_id, entry in spec["profiles"].items():
        pilot_id = entry["pilot_id"]
        candidate = candidates.get(pilot_id)
        if candidate is None:
            raise RuntimeError(f"runtime pilot_id is not selected: {pilot_id}")
        result = {
            "pilot_id": pilot_id,
            "source_point": candidate["source_point"],
            "program_checks": _program_checks(candidate),
            "rejected_pixel_attempts": [],
            "infrastructure_failures": list(
                entry.get("infrastructure_failures", [])),
        }
        for path in entry.get("rejected_pixel_joins", []):
            value = _read(path)
            if value.get("status") != "pixel_rejected":
                raise RuntimeError(f"failure ledger entry is not rejected: {path}")
            result["rejected_pixel_attempts"].append({
                "path": str(Path(path).resolve()),
                "rejection_reasons": value["rejection_reasons"],
            })
        if profile_id == "card17":
            visual1 = _visual_receipt(entry["visual_segment1"])
            visual2 = _visual_receipt(entry["visual_segment2"])
            result["visual_segment1"] = visual1
            result["visual_segment2"] = visual2
            result["segment_checks"] = _card17_distinct(
                entry["visual_segment1"], entry["visual_segment2"])
            result["audio_main"] = _audio_receipt(entry["audio_main"])
        else:
            result["visual"] = _visual_receipt(entry["visual"])
            result["pixel_join"] = _pixel_join(
                entry["pixel_join"], candidate["artifacts"]["fact"])
            result["audio_main"] = _audio_receipt(entry["audio_main"])
            result["audio_gatea"] = _audio_receipt(entry["audio_gatea"])
            result["audio_mixtures_differ"] = (
                result["audio_main"]["mixture_sha256"]
                != result["audio_gatea"]["mixture_sha256"])
            if not result["audio_mixtures_differ"]:
                raise RuntimeError(
                    f"{profile_id}: main and Gate-A mixtures are identical")
        results[profile_id] = result
    return {
        "schema": "qa_v3_room_pilot_runtime_manifest_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "boundary": (
            "Representative Apartment runtime evidence for the selected "
            "room-centric pilot; not full-pilot rendering, modality "
            "certification, human answerability, or formal admission."),
        "pilot_manifest": spec["pilot_manifest"],
        "profiles": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-manifest", required=True, type=Path)
    parser.add_argument("--runtime-spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output_root.exists() or args.output_root.is_symlink():
        print(f"refusing to overwrite: {args.output_root}", file=sys.stderr)
        return 2
    pilot = _read(args.pilot_manifest)
    spec = _read(args.runtime_spec)
    if Path(spec["pilot_manifest"]).resolve() != args.pilot_manifest.resolve():
        raise RuntimeError("runtime spec points to a different pilot manifest")
    result = finalize(pilot, spec)
    args.output_root.mkdir(parents=True)
    output = args.output_root / "pilot_runtime_manifest.json"
    _write(output, result)
    print(json.dumps({
        "output": str(output),
        "profiles": sorted(result["profiles"]),
        "status": result["status"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
