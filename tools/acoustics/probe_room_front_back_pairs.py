#!/usr/bin/env python3
"""Measure whether one room supports front/back mirrored source pairs.

A front/back question asks the listener to tell azimuth ``t`` from ``180 - t``.
That pair is special and hard for one reason: both members project the same
distance onto the interaural axis, so the interaural time difference is
identical and carries no information at all.  Whatever separates them has to be
the spectral shape the pinna imposes, which lives in roughly 2-8 kHz.  Whether
a real room preserves that shape or drowns it in early reflections cannot be
reasoned out -- reflections may fill the pinna notches or deepen them, and which
one happens depends on the room -- so it has to be measured, per room, before a
question family is built on it.

Three measurement decisions are worth stating, because the obvious versions of
all three give the wrong answer:

  * **Window to the direct arrival.**  Measured over a whole room impulse
    response the statistic saturates: in one apartment a 3 degree nudge and a
    120 degree move both landed near 7.6 dB, because the reverberant tail
    dominates the energy and the tail is diffuse and carries no direction.  The
    pinna response is over quickly -- about 97% of the energy of a KEMAR HRIR is
    inside 2 ms -- so a window of about that length holds the cue and admits the
    least room.  Longer windows add early reflections, which are real but are
    not specific to front and back.

  * **Compare against a pair that is separated by the same angle but is not a
    mirror.**  The tempting control is a small nudge, but the mirror pair's two
    points are metres apart in the room while a nudge moves centimetres, so the
    ratio flatters the mirror pair for a reason that has nothing to do with
    front and back.  For a front member ``t`` this tool also renders
    ``3t - 180``, which is the same angular distance from ``t`` as the mirror is
    but on the other side, and so is separated by a large, usable ITD.  It is
    the "what a difference we can definitely hear looks like" yardstick.

  * **Check the listener is actually indoors.**  A search for "every direction
    is audible" walks straight out of the building, where nothing blocks
    anything: the winning position in the first apartment sweep turned out to be
    in open space, and gave itself away by being dry.  Late energy measured
    against the direct arrival separates the two -- indoors the ratio sat near
    -1 to 0 dB, in the open it fell to -4 dB and below.

Nothing here is specific to a room, a listener, an azimuth or a distance: every
one of those is an argument.  In particular a pair does not need any specific
distance and does not need a full circle of clear directions around the
listener; it needs its two members at one distance, both audible.  Requiring
more than that rejects rooms for no reason.

Example:

    python tools/acoustics/probe_room_front_back_pairs.py \\
        --config <render dependencies>.json \\
        --listener-from-m1-request <m1 capture request>.json \\
        --source-height-m 0.721 --azimuth-deg 30 --distance-m 2.0 \\
        --output <fresh directory>

Research candidate only.  A passing room says the cue survives the room, which
is not the same as saying a listener or a model can use it; that question
belongs to a calibration pack, not to acoustics.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path


SCHEMA = "avengine_room_front_back_pair_probe_v1"
CONFIG_KEYS = (
    "repo",
    "package_manifest",
    "simulation_request",
    "hrtf",
    "runtime_prefix",
    "rlr_sdk_root",
)


class FrontBackProbeError(RuntimeError):
    """Raised when the probe cannot produce a measurement it can stand behind."""


def wrap_deg(value: float) -> float:
    """Fold an angle into (-180, 180]."""

    folded = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if folded == -180.0 else folded


def mirror_azimuth_deg(azimuth_deg: float) -> float:
    """The front/back partner of ``azimuth_deg``: same interaural projection."""

    return wrap_deg(180.0 - float(azimuth_deg))


def equal_separation_reference_deg(azimuth_deg: float) -> float:
    """A partner as far from ``azimuth_deg`` as the mirror is, on the other side.

    The mirror sits ``180 - 2t`` away from ``t``.  Stepping the same distance the
    other way gives ``t - (180 - 2t) = 3t - 180``, which has a large interaural
    time difference and so is a pair anything can separate.  It is the reference
    that tells a real front/back cue apart from "these two points are simply far
    apart in the room".
    """

    return wrap_deg(3.0 * float(azimuth_deg) - 180.0)


def azimuth_label(azimuth_deg: float) -> str:
    """A source id the runtime accepts: ASCII letters, digits, '.', '_' and '-'."""

    rounded = int(round(wrap_deg(azimuth_deg)))
    return "az%s%03d" % ("n" if rounded < 0 else "p", abs(rounded))


def listener_basis(rotation_xyzw):
    """Forward and right unit vectors for a yaw-only orientation, y up.

    Habitat points a camera down its local -Z, and +X is to its right, so these
    are the two vectors an azimuth is measured against.  Azimuth is right
    positive here, matching ``relative_azimuth_deg`` on the production side;
    the publication sign convention is applied at the publication edge, never
    here.
    """

    qx, qy, qz, qw = (float(v) for v in rotation_xyzw)
    if abs(qx) > 1.0e-6 or abs(qz) > 1.0e-6:
        raise FrontBackProbeError(
            "listener orientation is not yaw only; this probe places sources on "
            f"the horizontal circle and cannot honour roll or pitch: {rotation_xyzw}"
        )
    yaw = 2.0 * math.atan2(qy, qw)
    forward = (-math.sin(yaw), 0.0, -math.cos(yaw))
    right = (math.cos(yaw), 0.0, -math.sin(yaw))
    return yaw, forward, right


def place_source(listener_xyz, forward, right, azimuth_deg, distance_m, height_m):
    """World position of a source at one azimuth and horizontal distance."""

    radians = math.radians(float(azimuth_deg))
    return [
        listener_xyz[0] + distance_m * (math.cos(radians) * forward[0]
                                        + math.sin(radians) * right[0]),
        float(height_m),
        listener_xyz[2] + distance_m * (math.cos(radians) * forward[2]
                                        + math.sin(radians) * right[2]),
    ]


def load_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [key for key in CONFIG_KEYS if not config.get(key)]
    if missing:
        raise FrontBackProbeError(
            f"render config {path} is missing required keys: {sorted(missing)}"
        )
    return config


def listener_from_m1_request(path: Path):
    """Read a camera-colocated listener pose out of an M1 capture request."""

    request = json.loads(Path(path).read_text(encoding="utf-8"))
    rig = request.get("primary_camera_rig") or {}
    world_from_rig = rig.get("world_from_rig") or {}
    translation = world_from_rig.get("translation_m")
    rotation = world_from_rig.get("rotation_xyzw")
    if translation is None or rotation is None:
        raise FrontBackProbeError(
            f"{path} carries no primary_camera_rig.world_from_rig pose"
        )
    listener = request.get("listener") or {}
    offset = (listener.get("rig_from_listener") or {}).get("translation_m")
    if offset is not None and any(abs(float(v)) > 1.0e-9 for v in offset):
        raise FrontBackProbeError(
            f"{path} offsets the listener from the camera rig by {offset}; this "
            "probe measures at the rig pose and will not silently ignore that"
        )
    return ([float(v) for v in translation], [float(v) for v in rotation],
            str(request.get("request_id") or Path(path).parent.name))


def onset_index(ir, threshold_fraction: float) -> int:
    """First sample where either ear crosses a fraction of the loudest sample.

    The window has to start at the arrival rather than at sample zero, because
    the two members of a pair are rarely equidistant to the sample and a fixed
    start would compare the head of one response against the tail of the other.
    """

    import numpy as np

    envelope = np.abs(ir).max(axis=0)
    peak = float(envelope.max())
    if peak <= 0.0:
        raise FrontBackProbeError("impulse response is silent; nothing to window")
    return int(np.argmax(envelope >= threshold_fraction * peak))


def late_to_direct_db(ir, sample_rate_hz, direct_ms, late_start_ms,
                      threshold_fraction) -> float:
    """Energy after ``late_start_ms`` against the direct arrival, in dB.

    High indoors, where a tail keeps arriving; low in the open, where it does
    not.  This is what stops a search for clear directions from selecting a
    spot outside the building.
    """

    import numpy as np

    start = onset_index(ir, threshold_fraction)
    direct_end = start + int(sample_rate_hz * direct_ms / 1000.0)
    late_start = start + int(sample_rate_hz * late_start_ms / 1000.0)
    direct = float(np.sum(np.asarray(ir)[:, start:direct_end] ** 2))
    late = float(np.sum(np.asarray(ir)[:, late_start:] ** 2))
    return 10.0 * math.log10(max(late, 1e-30) / max(direct, 1e-30))


def peak_dbfs(ir) -> float:
    import numpy as np

    return 20.0 * math.log10(max(float(np.abs(np.asarray(ir)).max()), 1e-12))


def band_spectrum_db(samples, sample_rate_hz, low_hz, high_hz):
    import numpy as np

    length = 1 << int(math.ceil(math.log2(max(len(samples), 64))))
    freqs = np.fft.rfftfreq(length, 1.0 / sample_rate_hz)
    magnitude = np.abs(np.fft.rfft(samples, length))
    selected = (freqs >= low_hz) & (freqs <= high_hz)
    if not selected.any():
        raise FrontBackProbeError(
            f"band {low_hz}-{high_hz} Hz holds no bin at {sample_rate_hz} Hz"
        )
    return 20.0 * np.log10(np.maximum(magnitude[selected], 1e-12))


def spectral_difference_db(first, second, sample_rate_hz, low_hz, high_hz,
                           window_ms, threshold_fraction):
    """RMS difference of two responses' band spectra, overall level removed.

    Level is removed because it is a distance and occlusion cue, not a pinna
    cue, and the question here is only whether the *shape* differs.  ``None``
    for ``window_ms`` measures the whole response, which is what saturates.
    """

    import numpy as np

    per_ear = []
    for ear in (0, 1):
        left = np.asarray(first)[ear]
        right = np.asarray(second)[ear]
        if window_ms is not None:
            span = int(sample_rate_hz * window_ms / 1000.0)
            left_start = onset_index(first, threshold_fraction)
            right_start = onset_index(second, threshold_fraction)
            left = left[left_start:left_start + span]
            right = right[right_start:right_start + span]
        difference = (band_spectrum_db(left, sample_rate_hz, low_hz, high_hz)
                      - band_spectrum_db(right, sample_rate_hz, low_hz, high_hz))
        difference = difference - difference.mean()
        per_ear.append(float(np.sqrt(np.mean(difference ** 2))))
    return {"left": per_ear[0], "right": per_ear[1],
            "mean": (per_ear[0] + per_ear[1]) / 2.0}


def render_azimuths(runtime, listener_xyz, rotation_xyzw, azimuths_deg,
                    distance_m, source_height_m):
    """Render every azimuth through ONE context, so the room cannot differ.

    Rendering the two members of a pair in separate contexts would let engine
    state differ between them, and the whole measurement is a difference
    between the two, so that difference has to be the only one.
    """

    import numpy as np

    build_grid = runtime["build_strided_review_keyframes"]
    render_sequence = runtime["render_research_review_binaural_rir_sequence"]
    _, forward, right = listener_basis(rotation_xyzw)
    trajectories, placed = {}, {}
    for azimuth in azimuths_deg:
        label = azimuth_label(azimuth)
        position = place_source(listener_xyz, forward, right, azimuth,
                                distance_m, source_height_m)
        trajectories[label] = [position]
        placed[label] = {"azimuth_deg": wrap_deg(azimuth), "world_xyz_m": position}
    qx, qy, qz, qw = (float(v) for v in rotation_xyzw)
    grid = build_grid(
        trajectories,
        visual_frame_rate_hz=runtime["visual_frame_rate_hz"],
        rir_stride_frames=1,
        listener_position_m=list(listener_xyz),
        listener_orientation_wxyz=[qw, qx, qy, qz],
    )
    sequence = render_sequence(
        runtime["scene"], runtime["simulation"], grid=grid,
        hrtf_file_path=str(runtime["hrtf"]),
    )
    samples = np.asarray(sequence.samples)[0]
    responses = {source_id: np.asarray(samples[index], dtype=np.float32)
                 for index, source_id in enumerate(sequence.source_ids)}
    return int(sequence.sample_rate_hz), responses, placed


def load_runtime(config, visual_frame_rate_hz):
    """Import the production audio runtime and compile the room once."""

    os.environ["AVENGINE_HABITAT_RUNTIME_PREFIX"] = str(config["runtime_prefix"])
    os.environ["AVENGINE_RLR_SDK_ROOT"] = str(config["rlr_sdk_root"])
    if config.get("magnum_python_site"):
        os.environ["AVENGINE_HABITAT_MAGNUM_PYTHON_SITE"] = str(
            config["magnum_python_site"]
        )
    repo = Path(config["repo"]).resolve(strict=True)
    sys.path.insert(0, str(repo / "src"))
    from avengine.acoustics.runtime import load_compiled_acoustic_scene
    from avengine.capture.acoustics import (
        build_strided_review_keyframes,
        render_research_review_binaural_rir_sequence,
    )
    from avengine.spatial_audio.current_request_pair_ir import _load_simulation_request

    _, simulation = _load_simulation_request(Path(config["simulation_request"]))
    scene = load_compiled_acoustic_scene(
        config["package_manifest"], allow_nonpassing_research_qa=True
    )
    return {
        "scene": scene,
        "simulation": simulation,
        "hrtf": Path(config["hrtf"]).resolve(strict=True),
        "build_strided_review_keyframes": build_strided_review_keyframes,
        "render_research_review_binaural_rir_sequence": (
            render_research_review_binaural_rir_sequence
        ),
        "visual_frame_rate_hz": visual_frame_rate_hz,
        "write_float32_wav": __import__(
            "avengine.spatial_audio.audio", fromlist=["write_float32_wav"]
        ).write_float32_wav,
    }


def probe_one(runtime, pose, distance_m, azimuth_deg, args, bands):
    """One listener, one distance, one front azimuth: the pair and its yardsticks."""

    listener_xyz, rotation_xyzw, pose_id = pose
    mirror = mirror_azimuth_deg(azimuth_deg)
    nudge = wrap_deg(azimuth_deg + args.nudge_deg)
    reference = equal_separation_reference_deg(azimuth_deg)
    reference_nudge = wrap_deg(reference + args.nudge_deg)
    wanted = [azimuth_deg, mirror, nudge, reference, reference_nudge]
    sample_rate, responses, placed = render_azimuths(
        runtime, listener_xyz, rotation_xyzw, wanted, distance_m,
        args.source_height_m,
    )
    labels = {name: azimuth_label(value) for name, value in (
        ("front", azimuth_deg), ("mirror", mirror), ("nudge", nudge),
        ("reference", reference), ("reference_nudge", reference_nudge))}

    audibility = {}
    for name, label in labels.items():
        ir = responses[label]
        audibility[name] = {
            "azimuth_deg": placed[label]["azimuth_deg"],
            "world_xyz_m": placed[label]["world_xyz_m"],
            "peak_dbfs": peak_dbfs(ir),
            "late_to_direct_db": late_to_direct_db(
                ir, sample_rate, args.direct_window_ms, args.late_start_ms,
                args.onset_threshold,
            ),
        }

    pair_audible = all(
        audibility[name]["peak_dbfs"] >= args.min_peak_dbfs
        for name in ("front", "mirror")
    )
    indoors = (audibility["front"]["late_to_direct_db"]
               >= args.min_late_to_direct_db)

    measurements = {}
    for low_hz, high_hz in bands:
        band_key = f"{low_hz:.0f}-{high_hz:.0f}Hz"
        entry = {}
        for name, (a, b) in (
            ("mirror_vs_front", ("front", "mirror")),
            ("nudge_vs_front", ("front", "nudge")),
            ("reference_vs_front", ("front", "reference")),
            ("nudge_vs_reference", ("reference", "reference_nudge")),
        ):
            entry[name] = spectral_difference_db(
                responses[labels[a]], responses[labels[b]], sample_rate,
                low_hz, high_hz, args.direct_window_ms, args.onset_threshold,
            )
        mirror_mean = entry["mirror_vs_front"]["mean"]
        reference_mean = entry["reference_vs_front"]["mean"]
        nudge_mean = entry["nudge_vs_front"]["mean"]
        entry["mirror_over_reference"] = (
            mirror_mean / reference_mean if reference_mean else None
        )
        entry["mirror_over_nudge"] = (
            mirror_mean / nudge_mean if nudge_mean else None
        )
        measurements[band_key] = entry

    return {
        "pose_id": pose_id,
        "listener_position_m": list(listener_xyz),
        "listener_orientation_xyzw": list(rotation_xyzw),
        "horizontal_distance_m": distance_m,
        "source_height_m": args.source_height_m,
        "elevation_deg": -math.degrees(
            math.atan2(listener_xyz[1] - args.source_height_m, distance_m)
        ),
        "azimuths_deg": {name: placed[label]["azimuth_deg"]
                         for name, label in labels.items()},
        "sample_rate_hz": sample_rate,
        "audibility": audibility,
        "pair_audible": pair_audible,
        "listener_indoors": indoors,
        "usable": bool(pair_audible and indoors),
        "bands": measurements,
    }, sample_rate, responses, labels


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="JSON naming the render dependencies: repo, package_manifest, "
             "simulation_request, hrtf, runtime_prefix, rlr_sdk_root and "
             "optionally magnum_python_site. One per room package.",
    )
    listener = parser.add_argument_group("listener pose (choose one)")
    listener.add_argument(
        "--listener-from-m1-request", type=Path, action="append", default=[],
        help="read the camera-colocated pose out of an M1 capture request; "
             "repeatable",
    )
    listener.add_argument(
        "--listener-from-m1-glob",
        help="glob of M1 capture requests, to survey many poses at once",
    )
    listener.add_argument(
        "--listener", nargs=3, type=float, metavar=("X", "Y", "Z"),
        help="explicit listener position in room coordinates, metres",
    )
    listener.add_argument(
        "--listener-quat-xyzw", nargs=4, type=float,
        metavar=("X", "Y", "Z", "W"),
        help="explicit yaw-only listener orientation; with --listener",
    )
    parser.add_argument(
        "--source-height-m", required=True, type=float,
        help="emitter height in room coordinates. A room and asset property, "
             "so it is never defaulted: a dog and a speaker do not agree.",
    )
    parser.add_argument(
        "--azimuth-deg", type=float, action="append", default=[],
        help="front member of a mirrored pair, right positive from the "
             "listener's forward vector; repeatable. Default 30.",
    )
    parser.add_argument(
        "--distance-m", type=float, action="append", default=[],
        help="horizontal source distance; repeatable. A pair only needs its "
             "two members at one distance, so try what the room affords rather "
             "than insisting on a figure. Default 2.0.",
    )
    parser.add_argument(
        "--band-hz", nargs=2, type=float, action="append", default=[],
        metavar=("LOW", "HIGH"),
        help="analysis band; repeatable. Default 2000 6000, which sits below "
             "the reconstruction rolloff of a 16 kHz render while still "
             "holding most of the pinna response.",
    )
    parser.add_argument(
        "--direct-window-ms", type=float, default=2.0,
        help="length of the window from the direct arrival. About 97 percent "
             "of a KEMAR HRIR's energy lands inside 2 ms; longer windows admit "
             "early reflections, which are real but not specific to front and "
             "back. Pass 0 to measure the whole response, which saturates.",
    )
    parser.add_argument(
        "--late-start-ms", type=float, default=20.0,
        help="where the late field is taken to begin, for the indoor test",
    )
    parser.add_argument(
        "--onset-threshold", type=float, default=0.2,
        help="fraction of the loudest sample that counts as the arrival",
    )
    parser.add_argument(
        "--nudge-deg", type=float, default=3.0,
        help="small angular control, to show what a mere position change costs",
    )
    parser.add_argument(
        "--min-peak-dbfs", type=float, default=-45.0,
        help="a pair member quieter than this is behind something; the pair is "
             "reported as not usable rather than measured",
    )
    parser.add_argument(
        "--min-late-to-direct-db", type=float, default=-6.0,
        help="below this the listener is not in a room. Guards the case where "
             "a search for clear directions wanders outside the building, "
             "where nothing blocks anything and every direction passes.",
    )
    parser.add_argument(
        "--visual-frame-rate-hz", type=float, default=15.0,
        help="frame rate the keyframe grid is built against; static sources "
             "make this a formality, but it is declared rather than assumed",
    )
    parser.add_argument("--output", required=True, type=Path,
                        help="fresh output directory for the report and RIRs")
    parser.add_argument("--write-rirs", action="store_true",
                        help="also write each rendered response as float32 WAV")
    parser.add_argument("--resume", action="store_true",
                        help="allow an existing output directory")
    return parser.parse_args(argv)


def collect_poses(args):
    poses = []
    for path in args.listener_from_m1_request:
        poses.append(listener_from_m1_request(path))
    if args.listener_from_m1_glob:
        matched = sorted(glob.glob(args.listener_from_m1_glob))
        if not matched:
            raise FrontBackProbeError(
                f"--listener-from-m1-glob matched nothing: "
                f"{args.listener_from_m1_glob}"
            )
        for path in matched:
            poses.append(listener_from_m1_request(Path(path)))
    if args.listener is not None:
        if args.listener_quat_xyzw is None:
            raise FrontBackProbeError(
                "--listener needs --listener-quat-xyzw; an orientation cannot "
                "be guessed and a wrong one silently rotates every azimuth"
            )
        poses.append((list(args.listener), list(args.listener_quat_xyzw),
                      "explicit"))
    if not poses:
        raise FrontBackProbeError(
            "no listener pose given; pass --listener-from-m1-request, "
            "--listener-from-m1-glob or --listener with --listener-quat-xyzw"
        )
    return poses


def main(argv=None) -> int:
    args = parse_args(argv)
    azimuths = args.azimuth_deg or [30.0]
    distances = args.distance_m or [2.0]
    bands = [tuple(band) for band in args.band_hz] or [(2000.0, 6000.0)]
    if args.direct_window_ms == 0:
        args.direct_window_ms = None

    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FrontBackProbeError(
            f"{output} exists; this probe writes fresh output. Pass --resume "
            "only when you mean to add to an earlier run."
        )
    output.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    poses = collect_poses(args)
    runtime = load_runtime(config, args.visual_frame_rate_hz)

    rows = []
    for pose in poses:
        for distance in distances:
            for azimuth in azimuths:
                row, sample_rate, responses, labels = probe_one(
                    runtime, pose, distance, azimuth, args, bands
                )
                rows.append(row)
                verdict = "usable" if row["usable"] else (
                    "pair blocked" if not row["pair_audible"] else "not indoors")
                band_key = f"{bands[0][0]:.0f}-{bands[0][1]:.0f}Hz"
                entry = row["bands"][band_key]
                ratio = entry["mirror_over_reference"]
                print(
                    "%-14s %4.1f m  az %6.1f  %-12s  mirror %5.2f dB  "
                    "reference %5.2f dB  mirror/reference %s"
                    % (row["pose_id"], distance, wrap_deg(azimuth), verdict,
                       entry["mirror_vs_front"]["mean"],
                       entry["reference_vs_front"]["mean"],
                       "n/a" if ratio is None else "%.2f" % ratio),
                    flush=True,
                )
                if args.write_rirs:
                    stem = "%s_d%02d_%s" % (
                        row["pose_id"], int(round(distance * 10)),
                        azimuth_label(azimuth))
                    for name, label in labels.items():
                        runtime["write_float32_wav"](
                            output / f"rir_{stem}_{name}_{label}.wav",
                            responses[label], sample_rate,
                        )

    report = {
        "schema": SCHEMA,
        "usage_scope": "research_candidate",
        "config": str(Path(args.config).resolve()),
        "package_manifest": str(config["package_manifest"]),
        "hrtf": str(config["hrtf"]),
        "parameters": {
            "azimuths_deg": azimuths,
            "distances_m": distances,
            "bands_hz": [list(band) for band in bands],
            "source_height_m": args.source_height_m,
            "direct_window_ms": args.direct_window_ms,
            "late_start_ms": args.late_start_ms,
            "onset_threshold": args.onset_threshold,
            "nudge_deg": args.nudge_deg,
            "min_peak_dbfs": args.min_peak_dbfs,
            "min_late_to_direct_db": args.min_late_to_direct_db,
        },
        "azimuth_convention": (
            "right positive from the listener forward vector, engine frame"
        ),
        "rows": rows,
    }
    (output / "front_back_pair_probe.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    usable = sum(1 for row in rows if row["usable"])
    print("\nFRONT_BACK_PAIR_PROBE_OK rows=%d usable=%d output=%s"
          % (len(rows), usable, output / "front_back_pair_probe.json"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FrontBackProbeError as error:
        print(f"FRONT_BACK_PAIR_PROBE_FAILED {error}", file=sys.stderr)
        sys.exit(1)
