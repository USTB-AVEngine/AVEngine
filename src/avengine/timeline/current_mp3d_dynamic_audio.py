"""Motion-following research audio for authored current routes.

The static ``render_current_m1_research_audio`` verb stays the frozen
baseline: its pair IRs carry one state per source, so mixes hold no motion.
This module renders the audio the visual capture actually shows instead:
captured per-frame emitter positions drive a strided keyframe grid through
the persistent-context M5.1 review renderer, and the dry buses come from an
M6 AudioProgram routing variant, so the probe sources take turns instead of
sharing one schedule. The core ``render_dynamic_research_audio`` is
room-agnostic; ``render_current_mp3d_dynamic_audio`` binds it to the current
MP3D capture record layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import sha256_file
from avengine.contracts.transforms import compose_transforms
from avengine.rooms.contracts import validate_capture_request
from avengine.acoustics.runtime import load_compiled_acoustic_scene
from avengine.spatial_audio.audio import write_float32_wav
from avengine.spatial_audio.current_request_pair_ir import _load_simulation_request
from avengine.capture.acoustics import (
    build_strided_review_keyframes,
    render_research_review_binaural_audio,
    render_research_review_binaural_rir_sequence,
)
from avengine.timeline.audio_render import assemble_audio_program_dry_buses
from avengine.m6.sources import (
    load_sound_asset_registry,
    load_source_endpoint_registry,
    sound_index,
)

CURRENT_MP3D_DYNAMIC_AUDIO_SCHEMA = "avengine_m5_current_mp3d_dynamic_audio_v1"
VISUAL_FRAME_RATE_HZ = 15
EPISODE_FRAME_COUNT = 75
AUDIO_SAMPLE_RATE_HZ = 16_000
EPISODE_SAMPLE_COUNT = 80_000
CLAIM_BOUNDARY = (
    "Research review audio only. Captured per-frame source positions and one "
    "AudioProgram routing variant drive strided per-state binaural RIRs; no "
    "dataset admission, qualification, or new gate is claimed."
)


class CurrentMP3DDynamicAudioError(RuntimeError):
    """Raised when the dynamic research-audio contract is violated."""


def _fresh_output(path: str | Path) -> Path:
    output = Path(path).resolve()
    if output.exists() or output.is_symlink():
        raise CurrentMP3DDynamicAudioError(f"refusing to replace output: {output}")
    output.mkdir(parents=True)
    return output


def load_captured_source_paths(
    visual_capture_dir: str | Path, source_ids: tuple[str, ...]
) -> dict[str, list[list[float]]]:
    """Read the per-frame emitter world positions captured by current-visual.

    Frame-record slot ``i`` maps to ``source_ids[i]``: the capture writes the
    actor emitters in authoring order, which matches the byte-canonical
    program candidates for the two-beagle route.
    """

    records_path = Path(visual_capture_dir).resolve() / "frame_records.json"
    if not records_path.is_file():
        raise CurrentMP3DDynamicAudioError(
            f"visual capture is missing frame_records.json: {records_path}"
        )
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != EPISODE_FRAME_COUNT:
        raise CurrentMP3DDynamicAudioError(
            "frame_records must carry exactly the 75 episode frames"
        )
    trajectories: dict[str, list[list[float]]] = {
        source_id: [] for source_id in source_ids
    }
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or frame.get("frame_index") != index:
            raise CurrentMP3DDynamicAudioError(
                "frame_records indices must be contiguous from zero"
            )
        positions = frame.get("source_positions_m")
        if not isinstance(positions, list) or len(positions) != len(source_ids):
            raise CurrentMP3DDynamicAudioError(
                "each frame must record one source position per program candidate"
            )
        for slot, source_id in enumerate(source_ids):
            point = [float(value) for value in positions[slot]]
            if len(point) != 3 or not all(np.isfinite(point)):
                raise CurrentMP3DDynamicAudioError(
                    "source positions must be finite 3-vectors"
                )
            trajectories[source_id].append(point)
    return trajectories


def listener_pose_from_m1_request(
    m1_request: Mapping[str, Any],
) -> tuple[list[float], list[float]]:
    """Compose the static camera-colocated listener pose (wxyz orientation)."""

    errors = validate_capture_request(m1_request)
    if errors:
        raise CurrentMP3DDynamicAudioError("; ".join(errors))
    rig = m1_request["primary_camera_rig"]
    listener = m1_request["listener"]
    world_from_listener = compose_transforms(
        rig["world_from_rig"], listener["rig_from_listener"]
    )
    x, y, z, w = world_from_listener["rotation_xyzw"]
    return list(world_from_listener["translation_m"]), [w, x, y, z]


def _asset_bindings(
    sounds: Mapping[str, Any],
    *,
    repository_root: Path,
    external_sound_asset_paths: Mapping[str, Path],
    required_sound_ids: set[str],
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    records = sound_index(sounds)
    for sound_id in sorted(required_sound_ids):
        record = records.get(sound_id)
        if record is None:
            raise CurrentMP3DDynamicAudioError(
                f"program sound is not registered: {sound_id}"
            )
        uri = str(record["dry_audio"]["uri"])
        if uri.startswith("repo://"):
            resolved = (repository_root / uri.removeprefix("repo://")).resolve()
        else:
            path = external_sound_asset_paths.get(sound_id)
            if path is None:
                raise CurrentMP3DDynamicAudioError(
                    f"program sound {sound_id} requires an explicit external dry path"
                )
            resolved = Path(path).resolve()
        if not resolved.is_file():
            raise CurrentMP3DDynamicAudioError(f"dry audio is missing: {resolved}")
        expected = str(record["dry_audio"]["sha256"])
        if sha256_file(resolved) != expected:
            raise CurrentMP3DDynamicAudioError(
                f"dry audio differs from the registry digest for {sound_id}"
            )
        result[sound_id] = {"path": str(resolved), "sha256": expected}
    return result


def _input_record(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def render_dynamic_research_audio(
    *,
    source_trajectories_m: Mapping[str, Sequence[Sequence[float]]],
    listener_position_m: Sequence[float],
    listener_orientation_wxyz: Sequence[float],
    simulation_request_path: str | Path,
    package_manifest_path: str | Path,
    audio_program_path: str | Path,
    source_endpoint_registry_path: str | Path,
    sound_asset_registry_path: str | Path,
    external_sound_asset_paths: Mapping[str, str | Path],
    hrtf_file_path: str | Path,
    output_path: str | Path,
    position_authority: str,
    listener_authority: str,
    rir_stride_frames: int = 3,
    variant_id: str = "A",
    hrtf_license_path: str | Path | None = None,
    extra_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Room-agnostic core: trajectories + program variant -> binaural episode."""

    output = _fresh_output(output_path)
    program_path = Path(audio_program_path).resolve()
    program = json.loads(program_path.read_text(encoding="utf-8"))
    endpoint_registry_path = Path(source_endpoint_registry_path).resolve()
    sound_registry_path = Path(sound_asset_registry_path).resolve()
    endpoints = load_source_endpoint_registry(endpoint_registry_path)
    sounds = load_sound_asset_registry(sound_registry_path)

    source_ids = tuple(
        str(value) for value in program.get("candidate_source_endpoint_ids") or ()
    )
    if len(source_ids) < 2:
        raise CurrentMP3DDynamicAudioError(
            "the program must carry at least two candidate source endpoints"
        )
    if set(source_ids) != set(source_trajectories_m):
        raise CurrentMP3DDynamicAudioError(
            "trajectory source IDs must equal the program candidates: "
            f"{sorted(source_trajectories_m)} != {sorted(source_ids)}"
        )
    timeline = program.get("timeline") or {}
    if (
        timeline.get("frame_count") != EPISODE_FRAME_COUNT
        or timeline.get("sample_rate_hz") != AUDIO_SAMPLE_RATE_HZ
        or timeline.get("sample_count") != EPISODE_SAMPLE_COUNT
    ):
        raise CurrentMP3DDynamicAudioError(
            "the program timeline must match the 75-frame research episode"
        )

    ordered_trajectories = {
        source_id: [list(map(float, point)) for point in source_trajectories_m[source_id]]
        for source_id in source_ids
    }
    grid = build_strided_review_keyframes(
        ordered_trajectories,
        visual_frame_rate_hz=VISUAL_FRAME_RATE_HZ,
        rir_stride_frames=rir_stride_frames,
        listener_position_m=list(listener_position_m),
        listener_orientation_wxyz=list(listener_orientation_wxyz),
    )
    _, simulation = _load_simulation_request(Path(simulation_request_path).resolve())
    scene = load_compiled_acoustic_scene(
        package_manifest_path, allow_nonpassing_research_qa=True
    )
    hrtf = Path(hrtf_file_path).resolve()
    sequence = render_research_review_binaural_rir_sequence(
        scene, simulation, grid=grid, hrtf_file_path=str(hrtf)
    )

    required_sounds = {
        str(event["sound_asset_id"]) for event in program.get("events") or ()
    }
    repository_root = Path(__file__).resolve().parents[3]
    bindings = _asset_bindings(
        sounds,
        repository_root=repository_root,
        external_sound_asset_paths={
            key: Path(value) for key, value in external_sound_asset_paths.items()
        },
        required_sound_ids=required_sounds,
    )
    assembly = assemble_audio_program_dry_buses(
        program,
        variant_id,
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
        asset_bindings=bindings,
    )
    dry_buses = assembly.dry_audio.buses
    stems, mixture = render_research_review_binaural_audio(
        dry_buses, sequence, grid=grid
    )

    audio_root = output / "audio"
    outputs: dict[str, str] = {}

    def _write(path: Path, samples: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_float32_wav(path, samples, AUDIO_SAMPLE_RATE_HZ)
        outputs[str(path.relative_to(output))] = sha256_file(path)

    for source_id in source_ids:
        dry = np.asarray(dry_buses[source_id], dtype=np.float64)[None, :]
        _write(audio_root / "dry" / f"{source_id}.wav", dry)
        _write(
            audio_root / "binaural" / f"{source_id}_stem.wav",
            stems[source_id].episode,
        )
    _write(audio_root / "binaural" / "mixture.wav", np.asarray(mixture))

    materialized = assembly.materialized_program
    inputs: dict[str, Any] = {
        "simulation_request": _input_record(simulation_request_path),
        "package_manifest": _input_record(package_manifest_path),
        "source_endpoint_registry": _input_record(endpoint_registry_path),
        "sound_asset_registry": _input_record(sound_registry_path),
        "hrtf": {
            **_input_record(hrtf),
            "license_path": (
                str(Path(hrtf_license_path).resolve())
                if hrtf_license_path is not None
                else None
            ),
        },
        "dry_assets": bindings,
    }
    if extra_inputs:
        inputs.update({key: value for key, value in extra_inputs.items()})

    receipt: dict[str, Any] = {
        "schema": CURRENT_MP3D_DYNAMIC_AUDIO_SCHEMA,
        "status": "pass",
        "claim_boundary": CLAIM_BOUNDARY,
        "research_only": True,
        "episode_counted": False,
        "formal_dataset_count": 0,
        "qualification_claim": False,
        "audio": {
            "sample_rate_hz": AUDIO_SAMPLE_RATE_HZ,
            "sample_count": EPISODE_SAMPLE_COUNT,
            "layout_type": "binaural",
            "channel_labels": ["left", "right"],
        },
        "audio_program": {
            "path": str(program_path),
            "program_id": program.get("program_id"),
            "revision": program.get("revision"),
            "mode": program.get("mode"),
            "variant_id": variant_id,
            "program_content_sha256": materialized.get("program_content_sha256"),
            "event_count": len(program.get("events") or ()),
        },
        "sources": {
            "source_ids": list(source_ids),
            "frame_count": EPISODE_FRAME_COUNT,
            "position_authority": position_authority,
        },
        "listener": {
            "position_m": list(listener_position_m),
            "orientation_wxyz": list(listener_orientation_wxyz),
            "authority": listener_authority,
        },
        "rir": {
            "stride_frames": rir_stride_frames,
            "keyframe_count": len(sequence.keyframe_samples),
            "keyframe_samples": list(sequence.keyframe_samples),
            "trajectory_sha256": sequence.trajectory_sha256,
        },
        "inputs": inputs,
        "outputs": outputs,
    }
    (output / "research_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def render_current_mp3d_dynamic_audio(
    *,
    visual_capture_dir: str | Path,
    m1_request_path: str | Path,
    simulation_request_path: str | Path,
    package_manifest_path: str | Path,
    audio_program_path: str | Path,
    source_endpoint_registry_path: str | Path,
    sound_asset_registry_path: str | Path,
    external_sound_asset_paths: Mapping[str, str | Path],
    hrtf_file_path: str | Path,
    output_path: str | Path,
    rir_stride_frames: int = 3,
    variant_id: str = "A",
    hrtf_license_path: str | Path | None = None,
) -> dict[str, Any]:
    """Render one motion-following binaural research episode for MP3D."""

    program = json.loads(
        Path(audio_program_path).resolve().read_text(encoding="utf-8")
    )
    source_ids = tuple(
        str(value) for value in program.get("candidate_source_endpoint_ids") or ()
    )
    trajectories = load_captured_source_paths(visual_capture_dir, source_ids)
    m1_request = json.loads(
        Path(m1_request_path).resolve().read_text(encoding="utf-8")
    )
    listener_position, listener_wxyz = listener_pose_from_m1_request(m1_request)
    return render_dynamic_research_audio(
        source_trajectories_m=trajectories,
        listener_position_m=listener_position,
        listener_orientation_wxyz=listener_wxyz,
        simulation_request_path=simulation_request_path,
        package_manifest_path=package_manifest_path,
        audio_program_path=audio_program_path,
        source_endpoint_registry_path=source_endpoint_registry_path,
        sound_asset_registry_path=sound_asset_registry_path,
        external_sound_asset_paths=external_sound_asset_paths,
        hrtf_file_path=hrtf_file_path,
        output_path=output_path,
        position_authority=(
            "current-visual frame_records per-frame source_positions_m"
        ),
        listener_authority=(
            "research M1 request primary_camera_rig composed with "
            "rig_from_listener"
        ),
        rir_stride_frames=rir_stride_frames,
        variant_id=variant_id,
        hrtf_license_path=hrtf_license_path,
        extra_inputs={
            "visual_capture_frame_records": _input_record(
                Path(visual_capture_dir).resolve() / "frame_records.json"
            ),
            "m1_request": _input_record(m1_request_path),
        },
    )
