"""Render a moving sound source in one of the renderer's two output layouts.

Ambisonics and binaural are two layouts of one render, not two jobs. Splitting
them into two scripts put two copies of the time-varying convolution in the tree
and they had already drifted: one grew an analysis-window sweep, the other grew
the left/right cardinal proof, and neither had the other's. So this is one tool
with --layout, and everything that does not depend on the layout is written
once.

  ambisonics  four channels in world axes. Carries the whole field, needs a
              decoder to listen to, and is what the per-frame direction check
              reads because the intensity vector needs the directional channels.
  binaural    two channels, [left, right], through an explicit HRTF. Listenable
              on headphones with front and back intact, and defined entirely by
              where the head points.

A moving source needs a response per frame, because the direct path length, the
arrival direction and the early reflections all change as it moves. The dry
signal is then convolved block by block with overlap-add, so each response's
tail carries into the next block instead of the reverberation restarting every
frame.

Layout-specific checks, each of which caught something real:

  * ambisonics sweeps the analysis window. The safe length has been revised
    three times as conditions got more honest - 12 ms read zero on JAEGER's
    reverb-free responses, 2 ms held in a stone hall, 1 ms held with the default
    acoustic material, and 0.25 ms is what survives real materials. Do not trust
    the default; sweep it, which costs nothing once the responses exist.
  * binaural proves the two horizontal cardinals before rendering anything. It
    runs at the most open point on the floor rather than at the clip's listener,
    because it validates the renderer, the HRTF and the channel order, and those
    are properties of the setup. Running it at the listener conflated the two
    and failed: that listener stood near a wall, the right probe was blocked at
    ear height while the navmesh called the floor beneath it walkable, both ears
    came back eleven decibels down, and the collapse read exactly like a swapped
    channel pair.

Head orientation matters only for binaural, and it is given as a direction
vector rather than an angle. An angle was tried and produced the failure the
check exists to catch: the video picks its aim where yaw parameterises
(sin, 0, -cos) while the look-at consuming it derives yaw as atan2(-x, -z), so
one number named two orientations sixty degrees apart. Read it from the video
manifest, which records the vector.
"""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np
import quaternion  # noqa: F401  before habitat_sim, per the SoundSpaces issue
import habitat_sim
from habitat_sim.sensor import AudioSensorSpec, RLRAudioPropagationChannelLayoutType

SPEED_OF_SOUND = 343.0
# SoundSpaces emits ACN [W, Y, Z, X]; these index habitat world X, Y, Z.
CH_X, CH_Y, CH_Z = 3, 1, 2
WINDOW_SWEEP_MS = (0.25, 0.5, 1.0, 2.0, 4.0, 12.0)


def look_at(direction, np_quaternion):
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    yaw = math.atan2(-d[0], -d[2])
    pitch = math.asin(float(np.clip(d[1], -1.0, 1.0)))
    qy = np_quaternion(math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)
    qx = np_quaternion(math.cos(pitch / 2), math.sin(pitch / 2), 0.0, 0.0)
    return qy * qx


def rotation_matrix(quat):
    w, x, y, z = quat.w, quat.x, quat.y, quat.z
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def build_simulator(args, bank, sample_rate):
    scene = bank["scene"]
    if Path(scene).name.endswith(".basis.glb"):
        raise SystemExit(
            "refusing a *.basis.glb: without a BasisImporter every texture "
            "fails to load and the first render segfaults. Use the "
            "uncompressed sibling glb."
        )
    use_materials = bool(args.dataset_config and args.scene_id)
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = args.scene_id if use_materials else scene
    if use_materials:
        backend.scene_dataset_config_file = str(args.dataset_config)
    backend.load_semantic_mesh = use_materials
    backend.enable_physics = False
    sim = habitat_sim.Simulator(
        habitat_sim.Configuration(backend, [habitat_sim.agent.AgentConfiguration()])
    )
    navmesh = bank.get("navmesh")
    if navmesh and not sim.pathfinder.is_loaded:
        # Querying an unloaded PathFinder segfaults rather than raising.
        sim.pathfinder.load_nav_mesh(navmesh)

    spec = AudioSensorSpec()
    spec.uuid = "audio_sensor"
    spec.enableMaterials = use_materials
    if args.layout == "ambisonics":
        spec.channelLayout.type = RLRAudioPropagationChannelLayoutType.Ambisonics
        spec.channelLayout.channelCount = 4
    else:
        spec.channelLayout.type = RLRAudioPropagationChannelLayoutType.Binaural
        spec.channelLayout.channelCount = 2
    spec.position = [0.0, 0.0, 0.0]
    spec.acousticsConfig.sampleRate = sample_rate
    spec.acousticsConfig.directSHOrder = 1
    spec.acousticsConfig.indirectSHOrder = 1
    spec.acousticsConfig.indirect = not args.direct_only
    sim.add_sensor(spec)
    sensor = sim.get_agent(0)._sensors["audio_sensor"]
    if args.layout == "binaural":
        sensor.setListenerHRTF(str(resolve_hrtf(args.hrtf)))
    if use_materials:
        if not args.materials_json:
            raise SystemExit(
                "--materials-json is required with --dataset-config; "
                "enableMaterials only tells the sensor to look for a database"
            )
        sensor.setAudioMaterialsJSON(str(args.materials_json))
    return sim, sensor, use_materials


def resolve_hrtf(path):
    if path is None:
        raise SystemExit("--hrtf is required for --layout binaural")
    path = Path(path)
    if path.is_dir():
        found = sorted(path.glob("*.sofa"))
        if not found:
            raise SystemExit(f"no .sofa under {path}")
        return found[0]
    return path


def onset(w, fraction=0.15):
    return int(np.argmax(np.abs(w) > fraction * np.max(np.abs(w)) + 1e-12))


def direct_direction(ir, window_ms, sample_rate):
    w = ir[0]
    start = onset(w)
    count = max(int(window_ms * 0.001 * sample_rate), 1)
    window = slice(start, start + count)
    reference = w[window]
    vector = np.array(
        [float(np.sum(reference * ir[c][window])) for c in (CH_X, CH_Y, CH_Z)]
    )
    norm = float(np.linalg.norm(vector))
    return (None if norm <= 0 else vector / norm), start


def t20_ms(w, sample_rate):
    peak = int(np.argmax(np.abs(w)))
    tail = np.asarray(w[peak:], dtype=float)
    if tail.size < 16:
        return float("nan")
    energy = np.cumsum(tail[::-1] ** 2)[::-1]
    curve = 10.0 * np.log10(np.maximum(energy / energy[0], 1e-20))
    a = np.flatnonzero(curve <= -5.0)
    b = np.flatnonzero(curve <= -25.0)
    if not a.size or not b.size:
        return float("nan")
    return float((b[0] - a[0]) / sample_rate * 1000.0)


def level_db(x):
    return 10.0 * math.log10(float(np.sum(np.asarray(x, dtype=float) ** 2)) + 1e-30)


def dry_signal(samples, seed, sample_rate):
    """Band-limited noise bursts.

    Synthesised rather than taken from a recording library: this is a
    localisation probe, sharp onsets are what make a moving direction audible,
    and it keeps the artifact free of any recording's licence.
    """

    rng = np.random.default_rng(seed)
    signal = np.zeros(samples, dtype=float)
    burst = int(0.05 * sample_rate)
    period = int(0.20 * sample_rate)
    envelope = np.hanning(burst)
    for start in range(0, samples - burst, period):
        noise = rng.normal(0.0, 1.0, burst)
        noise = np.convolve(np.diff(noise, prepend=0.0), np.ones(4) / 4.0, mode="same")
        signal[start : start + burst] += envelope * noise
    peak = float(np.max(np.abs(signal)))
    return signal / peak if peak > 0 else signal


def write_wave(path, channels, sample_rate):
    peak = float(np.max(np.abs(channels)))
    scaled = channels / peak * 0.97 if peak > 0 else channels
    data = (scaled.T * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels.shape[0])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(data.tobytes())


def convolve_route(responses, hop, sample_rate, seed, channels):
    total = hop * len(responses)
    dry = dry_signal(total, seed, sample_rate)
    tail = max(ir.shape[1] for ir in responses)
    wet = np.zeros((channels, total + tail), dtype=float)
    for position, ir in enumerate(responses):
        block = dry[position * hop : (position + 1) * hop]
        if not block.size:
            continue
        for channel in range(channels):
            piece = np.convolve(block, ir[channel])
            wet[channel, position * hop : position * hop + piece.size] += piece
    return wet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--layout", choices=("ambisonics", "binaural"), default="ambisonics"
    )
    parser.add_argument("--motion-case", default="source1_moving_source2_static")
    parser.add_argument("--slot", default="source1")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--episode-id")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--emitter-height-m", type=float)
    parser.add_argument(
        "--listener-pose",
        type=Path,
        help=(
            "pose file from tools/scene/choose_listener_pose.py. The preferred "
            "route: it carries both position and orientation, this pass "
            "auditions its candidates in order and writes accepted_index back, "
            "and the video pass then reads the same entry. Without it the two "
            "halves each decide half of one thing and the caller has to carry "
            "a bare vector between them"
        ),
    )
    parser.add_argument("--listener", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--listener-height-m", type=float, default=1.5)
    parser.add_argument("--listener-minimum-range-m", type=float, default=2.0)
    parser.add_argument("--listener-maximum-range-m", type=float, default=6.0)
    parser.add_argument("--listener-attempts", type=int, default=12)
    parser.add_argument("--from-report", type=Path,
                        help="reuse the listener and frames of an earlier render")
    parser.add_argument("--video-manifest", type=Path,
                        help="head orientation for binaural, as a vector")
    parser.add_argument("--aim", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--hrtf", type=Path)
    parser.add_argument("--dataset-config",
                        help="semantic scene dataset config; turns materials on")
    parser.add_argument("--scene-id")
    parser.add_argument("--materials-json",
                        help="acoustic material database, required with --dataset-config")
    parser.add_argument("--direct-only", action="store_true",
                        help="no indirect sound. Reproduces JAEGER's released RIRs")
    parser.add_argument("--direct-window-ms", type=float, default=0.25)
    parser.add_argument("--window-sweep", action="store_true")
    parser.add_argument("--tolerance-deg", type=float, default=5.0)
    parser.add_argument("--cardinal-margin-db", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    sample_rate = args.sample_rate
    earlier = (
        json.loads(args.from_report.read_text(encoding="utf-8"))
        if args.from_report
        else None
    )

    if args.episode_id:
        episode = next(e for e in bank["episodes"] if e["episode_id"] == args.episode_id)
    elif earlier:
        episode = next(
            e for e in bank["episodes"] if e["episode_id"] == earlier["episode_id"]
        )
    else:
        matching = [e for e in bank["episodes"] if e["motion_case"] == args.motion_case]
        if not matching:
            raise SystemExit(f"no {args.motion_case} episodes in {args.bank}")
        episode = matching[args.episode_index]

    slot = earlier["slot"] if earlier else args.slot
    floor_path = np.asarray(episode["source_center_paths_m"][slot], dtype=float)
    height = (
        float(args.emitter_height_m)
        if args.emitter_height_m is not None
        else float(
            earlier["source_center_height_m"]
            if earlier
            else bank["source_center_heights_m"][slot]
        )
    )
    emitters = floor_path.copy()
    emitters[:, 1] += height
    stride = earlier["frame_stride"] if earlier else args.frame_stride
    frames = (
        [f["frame"] for f in earlier["per_frame"]]
        if earlier
        else list(range(0, len(emitters), stride))
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sim, sensor, materials_on = build_simulator(args, bank, sample_rate)
    agent = sim.get_agent(0)
    np_quaternion = np.quaternion
    print(f"layout {args.layout}   materials {'on' if materials_on else 'off'}"
          f"   indirect {'off' if args.direct_only else 'on'}")

    pose = (
        json.loads(args.listener_pose.read_text(encoding="utf-8"))
        if args.listener_pose
        else None
    )

    aim = None
    if args.layout == "binaural":
        if pose is not None:
            index = pose.get("accepted_index")
            if index is None:
                index = 0
                print(
                    "the pose file has no accepted_index yet, so candidate 0 is "
                    "used. Run the ambisonic pass first and this follows its "
                    "acoustic audition instead of guessing"
                )
            aim = np.asarray(pose["candidates"][index]["aim_world"], dtype=float)
        elif args.video_manifest:
            aim = np.asarray(
                json.loads(args.video_manifest.read_text(encoding="utf-8"))[
                    "camera_aim_world"
                ],
                dtype=float,
            )
        elif args.aim:
            aim = np.asarray(args.aim, dtype=float)
        else:
            raise SystemExit(
                "binaural needs a head orientation: --video-manifest (preferred, "
                "it records the vector the picture used) or --aim"
            )
        aim = aim / np.linalg.norm(aim)
    rotation = (
        look_at(aim, np_quaternion) if aim is not None
        else np_quaternion(1.0, 0.0, 0.0, 0.0)
    )

    def place(position):
        state = agent.get_state()
        state.position = np.asarray(position, dtype=np.float32)
        state.rotation = rotation
        state.sensor_states = {}
        agent.set_state(state, True)

    def render(point):
        sensor.setAudioSourceTransform(np.asarray(point, dtype=np.float32))
        return np.asarray(
            sim.get_sensor_observations()["audio_sensor"], dtype=float
        )

    midpoint = emitters[frames[len(frames) // 2]]
    floor = float(bank["floor_height_m"])

    # --- the listener -----------------------------------------------------
    auditions = []
    listener = None
    accepted_index = None
    if pose is not None and args.layout == "binaural":
        index = pose.get("accepted_index") or 0
        listener = np.asarray(pose["candidates"][index]["position_m"], dtype=float)
        accepted_index = index
        print(f"listener from pose candidate {index}: "
              f"{np.round(listener, 3).tolist()}")
    elif pose is not None:
        # Try the ranked candidates in order and keep the first the sound agrees
        # with. Ranking is by openness and by how much of the route is in frame,
        # neither of which knows whether a wall stands between the two.
        for index, candidate in enumerate(pose["candidates"]):
            position = np.asarray(candidate["position_m"], dtype=float)
            place(position)
            measured, _ = direct_direction(
                render(midpoint), args.direct_window_ms, sample_rate
            )
            geometric = midpoint - position
            geometric = geometric / np.linalg.norm(geometric)
            error = (
                None if measured is None
                else float(
                    np.degrees(
                        np.arccos(np.clip(float(np.dot(measured, geometric)), -1, 1))
                    )
                )
            )
            auditions.append(
                {
                    "candidate_index": index,
                    "position_m": candidate["position_m"],
                    "midpoint_error_deg": None if error is None else round(error, 3),
                }
            )
            print(f"  pose candidate {index}  route in frame "
                  f"{100 * candidate['route_share_in_frame']:5.1f}%  "
                  f"midpoint error "
                  f"{'none' if error is None else f'{error:6.2f}'} deg", flush=True)
            if error is not None and error <= args.tolerance_deg:
                listener = position
                accepted_index = index
                break
        if listener is None:
            raise SystemExit(
                "no pose candidate could hear the route; widen the range band "
                "or ask choose_listener_pose.py for more candidates"
            )
        pose["accepted_index"] = accepted_index
        args.listener_pose.write_text(
            json.dumps(pose, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        print(f"accepted candidate {accepted_index}; written back to "
              f"{args.listener_pose.name}")
    elif earlier:
        listener = np.asarray(earlier["listener_m"], dtype=float)
        print(f"listener from {args.from_report.name}: "
              f"{np.round(listener, 3).tolist()}")
    elif args.listener:
        listener = np.asarray(args.listener, dtype=float)
        print(f"listener pinned to {np.round(listener, 3).tolist()}")
    elif args.layout == "binaural" and pose is None:
        raise SystemExit(
            "binaural cannot audition a listener: the audition scores candidates "
            "by the intensity vector, which needs the directional channels. "
            "Pass --listener, or --from-report to reuse an ambisonic run's choice"
        )
    else:
        rng = np.random.default_rng(args.seed)
        for _ in range(4000):
            candidate = np.asarray(
                sim.pathfinder.get_random_navigable_point(), dtype=float
            )
            if abs(candidate[1] - floor) > 0.5:
                continue
            candidate[1] = floor + args.listener_height_m
            span = float(np.linalg.norm(candidate - midpoint))
            if not (
                args.listener_minimum_range_m <= span <= args.listener_maximum_range_m
            ):
                continue
            place(candidate)
            measured, _ = direct_direction(
                render(midpoint), args.direct_window_ms, sample_rate
            )
            geometric = midpoint - candidate
            geometric = geometric / np.linalg.norm(geometric)
            error = (
                None if measured is None
                else float(
                    np.degrees(
                        np.arccos(np.clip(float(np.dot(measured, geometric)), -1, 1))
                    )
                )
            )
            auditions.append(
                {
                    "position_m": [round(float(v), 4) for v in candidate],
                    "range_to_midpoint_m": round(span, 3),
                    "midpoint_error_deg": None if error is None else round(error, 3),
                }
            )
            print(f"  audition {len(auditions):>2}  range {span:5.2f} m  "
                  f"midpoint error "
                  f"{'none' if error is None else f'{error:6.2f}'} deg", flush=True)
            if error is not None and error <= args.tolerance_deg:
                listener = candidate
                break
            if len(auditions) >= args.listener_attempts:
                break
        if listener is None:
            raise SystemExit(
                f"no listener passed the midpoint audition in {len(auditions)} "
                "attempts; every candidate in the band is screened from this route"
            )

    # --- binaural: prove left and right, away from the clip's listener ----
    cardinals = None
    if args.layout == "binaural":
        right_axis = rotation_matrix(rotation) @ np.array([1.0, 0.0, 0.0])
        print(f"head faces {np.round(aim, 3).tolist()}; "
              f"right is {np.round(right_axis, 3).tolist()}")
        best, best_clearance = None, -1.0
        for _ in range(600):
            candidate = np.asarray(
                sim.pathfinder.get_random_navigable_point(), dtype=float
            )
            if abs(candidate[1] - floor) > 0.5:
                continue
            clearance = float(
                sim.pathfinder.distance_to_closest_obstacle(candidate, 6.0)
            )
            if clearance > best_clearance:
                best, best_clearance = candidate, clearance
        if best is None:
            raise SystemExit("found no navigable point on this floor to check on")
        head = best.copy()
        head[1] = floor + height
        print(f"cardinal check at the most open point: "
              f"{np.round(head, 2).tolist()}, {best_clearance:.2f} m clearance")
        place(head)
        cardinals = {
            "check_position_m": [round(float(v), 3) for v in head],
            "check_clearance_m": round(best_clearance, 3),
        }
        reach = min(best_clearance * 0.6, 1.0)
        for side, sign in (("left", -1.0), ("right", 1.0)):
            ir = render(head + sign * right_axis * reach)
            left, right = level_db(ir[0]), level_db(ir[1])
            cardinals[side] = {
                "left_channel_db": round(left, 2),
                "right_channel_db": round(right, 2),
                "difference_db": round(left - right, 2),
            }
            print(f"  probe {side:<5} L {left:7.2f} dB  R {right:7.2f} dB  "
                  f"L-R {left - right:+7.2f} dB")
        if cardinals["left"]["difference_db"] < args.cardinal_margin_db:
            raise SystemExit(
                "a source on the head's left is not louder in the left channel "
                f"by {args.cardinal_margin_db} dB; refusing a mirrored clip"
            )
        if cardinals["right"]["difference_db"] > -args.cardinal_margin_db:
            raise SystemExit(
                "a source on the head's right is not louder in the right channel "
                f"by {args.cardinal_margin_db} dB; refusing a mirrored clip"
            )

    # --- the route --------------------------------------------------------
    place(listener)
    responses = []
    per_frame = []
    for frame in frames:
        emitter = emitters[frame]
        ir = render(emitter)
        responses.append(ir)
        geometric = emitter - listener
        distance = float(np.linalg.norm(geometric))
        entry = {"frame": frame, "range_m": round(distance, 3)}
        if args.layout == "ambisonics":
            measured, start = direct_direction(
                ir, args.direct_window_ms, sample_rate
            )
            expected = int(round(distance / SPEED_OF_SOUND * sample_rate))
            entry["onset_minus_direct_ms"] = round(
                (start - expected) / sample_rate * 1000.0, 3
            )
            if measured is None:
                entry["error_deg"] = None
            else:
                cosine = float(
                    np.clip(float(np.dot(measured, geometric / distance)), -1, 1)
                )
                entry["error_deg"] = round(float(np.degrees(np.arccos(cosine))), 3)
        else:
            entry["interaural_level_difference_db"] = round(
                level_db(ir[0]) - level_db(ir[1]), 2
            )
        per_frame.append(entry)

    channels = 4 if args.layout == "ambisonics" else 2
    hop = int(round(sample_rate / float(bank["frame_rate_hz"]))) * max(stride, 1)
    wet = convolve_route(responses, hop, sample_rate, args.seed, channels)
    stem = "ambisonic" if args.layout == "ambisonics" else "binaural"
    write_wave(args.output_dir / f"moving_source.{stem}.wav", wet, sample_rate)
    if args.layout == "ambisonics":
        # A cardioid pair on the left-right axis: audible anywhere, and lossy in
        # exactly the dimension ambisonics exists for, so front and back
        # collapse. Use --layout binaural to hear those.
        write_wave(
            args.output_dir / "moving_source.stereo_fold.wav",
            np.stack((wet[0] - wet[CH_X], wet[0] + wet[CH_X]), axis=0),
            sample_rate,
        )
    tail = max(ir.shape[1] for ir in responses)
    stacked = np.zeros((len(responses), channels, tail), dtype=np.float32)
    for position, ir in enumerate(responses):
        stacked[position, :, : ir.shape[1]] = ir
    np.save(args.output_dir / f"responses.{stem}.npy", stacked)

    decays = np.array(
        [t20_ms(ir[0], sample_rate) for ir in responses], dtype=float
    )
    decays = decays[np.isfinite(decays)]
    reverberation = {"t20_ms_median": None}
    if decays.size:
        reverberation = {
            "t20_ms_median": round(float(np.median(decays)), 2),
            "t20_ms_minimum": round(float(decays.min()), 2),
            "t20_ms_maximum": round(float(decays.max()), 2),
            "materials": materials_on,
        }
        print(f"T20 along the route  median {np.median(decays):.1f} ms  "
              f"({decays.min():.1f}-{decays.max():.1f})")

    sweep = None
    if args.window_sweep and args.layout == "ambisonics":
        sweep = {}
        print("\nwindow sweep over the responses already rendered")
        table = []
        for position, frame in enumerate(frames):
            geometric = emitters[frame] - listener
            distance = float(np.linalg.norm(geometric))
            geometric = geometric / distance
            row = []
            for window in WINDOW_SWEEP_MS:
                measured, _ = direct_direction(
                    responses[position], window, sample_rate
                )
                row.append(
                    float("nan") if measured is None
                    else float(
                        np.degrees(
                            np.arccos(
                                np.clip(float(np.dot(measured, geometric)), -1, 1)
                            )
                        )
                    )
                )
            table.append(row)
        table = np.asarray(table)
        for column, window in enumerate(WINDOW_SWEEP_MS):
            values = table[:, column]
            inside = int(np.count_nonzero(values <= args.tolerance_deg))
            sweep[f"{window}ms"] = {
                "median_deg": round(float(np.median(values)), 3),
                "maximum_deg": round(float(values.max()), 3),
                "frames_within_tolerance": inside,
            }
            print(f"  window {window:5.2f} ms   median {np.median(values):6.2f}  "
                  f"max {values.max():6.2f}  within {args.tolerance_deg:g} deg "
                  f"{inside}/{len(table)}")

    report = {
        "schema": "avengine_moving_source_render_v2",
        "layout": args.layout,
        "channels": channels,
        "sample_rate_hz": sample_rate,
        "scene": bank["scene"],
        "bank": str(args.bank),
        "episode_id": episode["episode_id"],
        "motion_case": episode["motion_case"],
        "slot": slot,
        "listener_m": [round(float(v), 4) for v in listener],
        "listener_pose": str(args.listener_pose) if args.listener_pose else None,
        "listener_pose_candidate": accepted_index,
        "listener_auditions": auditions,
        "source_center_height_m": height,
        "frames_rendered": len(frames),
        "frame_stride": stride,
        "seconds": round(hop * len(frames) / sample_rate, 3),
        "acoustic_materials": materials_on,
        "indirect": not args.direct_only,
        "reverberation": reverberation,
        "per_frame": per_frame,
    }
    if args.layout == "ambisonics":
        errors = np.array(
            [f["error_deg"] for f in per_frame if f.get("error_deg") is not None],
            dtype=float,
        )
        report["direct_window_ms"] = args.direct_window_ms
        report["tolerance_deg"] = args.tolerance_deg
        report["channel_order"] = "soundspaces ACN [W, Y, Z, X]"
        if errors.size:
            report["direction_error_deg"] = {
                "median": round(float(np.median(errors)), 3),
                "p90": round(float(np.percentile(errors, 90)), 3),
                "maximum": round(float(errors.max()), 3),
            }
            report["frames_within_tolerance"] = int(
                np.count_nonzero(errors <= args.tolerance_deg)
            )
            print(f"DoA error  median {np.median(errors):.2f}  "
                  f"p90 {np.percentile(errors, 90):.2f}  "
                  f"max {errors.max():.2f} deg")
            print(f"within {args.tolerance_deg:g} deg: "
                  f"{report['frames_within_tolerance']}/{errors.size} frames")
        if sweep:
            report["window_sweep"] = sweep
    else:
        hrtf = resolve_hrtf(args.hrtf)
        report["hrtf"] = str(hrtf)
        report["hrtf_provenance_present"] = (hrtf.parent / "PROVENANCE.json").is_file()
        report["hrtf_licence_present"] = (hrtf.parent / "LICENSE.txt").is_file()
        report["channel_order"] = "[left, right]"
        report["head_aim_world"] = [round(float(v), 6) for v in aim]
        report["cardinal_probes"] = cardinals
        report["cardinal_margin_db"] = args.cardinal_margin_db
        ild = np.array(
            [f["interaural_level_difference_db"] for f in per_frame], dtype=float
        )
        report["interaural_level_difference_db"] = {
            "minimum": round(float(ild.min()), 2),
            "median": round(float(np.median(ild)), 2),
            "maximum": round(float(ild.max()), 2),
        }
        print(f"interaural level difference {ild.min():+.1f} to {ild.max():+.1f} dB "
              f"(median {np.median(ild):+.1f})")

    (args.output_dir / "render_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output_dir}")
    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
