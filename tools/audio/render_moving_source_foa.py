"""Render a moving sound source along a route the navmesh already approved.

A static source needs one impulse response. A moving one needs a new response
at every frame, because the direct path length, the arrival direction and the
early reflections all change as it walks. So this renders one FOA response per
frame at the emitter's position for that frame, verifies each one against the
geometry it should encode, and then convolves a dry signal block by block with
overlap-add so consecutive frames use consecutive responses. That is what makes
the result a moving source rather than a static one with the level swept.

The per-frame direction check is the whole point. A route can be perfectly
walkable and still have stretches where the listener cannot hear where the
source is, because a wall is in the way. Averaging over the route would hide
exactly that, so the error is reported per frame and the occluded stretches are
named.

The listener needs the same treatment, and it was the weak link. Sampling a
navigable point inside a plausible distance band puts it behind a wall about
half the time: over six episodes in one scene, three verified every frame
within 0.27 degrees and three failed every frame, with medians of 44, 44 and 70
and one frame reading 171 - the intensity vector pointing away from the source
because the only energy arriving came off a wall. A geodesic distance cannot
predict this, because the navmesh knows nothing about what stands at ear
height. Only the render can, so the listener is chosen by rendering: one
response at the route midpoint per candidate, which costs a second or two,
before committing to the full per-frame pass.

Two decisions worth stating rather than leaving in the code:

  * the analysis window defaults to 0.25 ms, and that number has now been
    revised three times, each time because the test conditions got more honest.
    12 ms looked perfect on JAEGER's released responses, which carry no
    reverberation at all, so any window returned zero. 2 ms looked safe on
    skokloster-castle, a large stone hall. 1 ms held in a furnished house
    rendered with the default acoustic material. With HM3D semantics supplying
    real materials it fails too: swept over four routes, medians at 1 ms come
    out 0.38, 7.25, 0.31 and 6.06 degrees, while 0.25 ms gives 0.00, 0.01, 0.05
    and 0.00.

    Note the direction, because it is not the obvious one. Materials *reduce*
    reverberation - T20 on the same route and listener falls from 171.6 ms to
    81.0 - and the safe window still gets *shorter*. Late energy is not what
    contaminates a 1 ms window; the early reflections are, and assigning real
    materials changes which of them land inside it.

    So do not trust this constant either. Pass --window-sweep, which costs
    nothing because the responses are already rendered, and read the table for
    the room you are actually in. And note what 0.25 ms means at 16 kHz: four
    samples. It works because the direct arrival is nearly impulsive, and it
    will not survive a source whose onset is soft.
  * the route's own samples sit on the floor. The emitter is the source centre,
    so its height is added on top; the bank records that height, and using the
    floor points directly would put a walking source's mouth on the carpet.

This runs on the SoundSpaces habitat build, not the AVEngine runtime prefix.
The prefix carries the AudioSensorSpec class but its
RLRAudioPropagationChannelLayoutType is None, so the upstream audio sensor path
is not wired there. AVEngine's own acoustic runtime is the eventual home, but
simulate_compiled_acoustic_scene wants a CompiledAcousticScene - an M3 package
with a material database and per-material triangle counts - and compiling an
HM3D scan into one means first deciding its acoustic materials, which the raw
glb carries no annotations for.
"""

from __future__ import annotations  # ss2 runs Python 3.9; str | None needs this

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np
import quaternion  # noqa: F401  imported before habitat_sim per SoundSpaces issue
import habitat_sim
from habitat_sim.sensor import AudioSensorSpec, RLRAudioPropagationChannelLayoutType

SR = 16000
SPEED_OF_SOUND = 343.0
# SoundSpaces 2.0 emits ACN [W, Y, Z, X]; these index habitat world X, Y, Z.
CH_X, CH_Y, CH_Z = 3, 1, 2


def make_sim(
    scene: str,
    navmesh: str | None,
    dataset_config: str | None = None,
    scene_id: str | None = None,
    materials_json: str | None = None,
):
    """Build the simulator, with acoustic materials when a dataset config is given.

    Materials need three things at once and each one fails quietly on its own:
    the scene opened through a semantic scene dataset config rather than as a
    bare glb path, load_semantic_mesh on, and the material database handed to
    the sensor explicitly - enableMaterials only says "look for one".

    The config has to be the vertex-colour variant. HM3D-Semantics v0.2 ships
    has_semantic_textures true, and this habitat build understands only
    vertex-coloured semantics, so it reports "Mesh vertices were NULL", uploads
    no geometry, and silently returns a direct-path-only response. The v0.2
    .semantic.glb does carry COLOR_0 alongside its textures, so flipping that
    flag to false in a copy of the config is all it takes.
    """

    use_materials = bool(dataset_config and scene_id)
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene_id if use_materials else scene
    if use_materials:
        backend.scene_dataset_config_file = str(dataset_config)
    backend.load_semantic_mesh = use_materials
    backend.enable_physics = False
    sim = habitat_sim.Simulator(
        habitat_sim.Configuration(backend, [habitat_sim.agent.AgentConfiguration()])
    )
    if navmesh and not sim.pathfinder.is_loaded:
        # Querying an unloaded PathFinder segfaults rather than raising.
        sim.pathfinder.load_nav_mesh(navmesh)
    audio = AudioSensorSpec()
    audio.uuid = "audio_sensor"
    audio.enableMaterials = use_materials
    audio.channelLayout.type = RLRAudioPropagationChannelLayoutType.Ambisonics
    audio.channelLayout.channelCount = 4
    audio.position = [0.0, 0.0, 0.0]
    audio.acousticsConfig.sampleRate = SR
    audio.acousticsConfig.directSHOrder = 1
    audio.acousticsConfig.indirectSHOrder = 1
    audio.acousticsConfig.indirect = True
    sim.add_sensor(audio)
    if use_materials and materials_json:
        sim.get_agent(0)._sensors["audio_sensor"].setAudioMaterialsJSON(
            str(materials_json)
        )
    return sim


def direct_arrival_direction(ir, distance_m, window_ms):
    """Direction of the direct path, windowed where geometry says it arrives."""

    w = ir[0]
    threshold = 0.15 * np.max(np.abs(w)) + 1e-12
    detected = int(np.argmax(np.abs(w) > threshold))
    expected = int(round(distance_m / SPEED_OF_SOUND * SR))
    window = slice(detected, detected + max(int(window_ms * 0.001 * SR), 1))
    reference = w[window]
    vector = np.array(
        [
            float(np.sum(reference * ir[CH_X][window])),
            float(np.sum(reference * ir[CH_Y][window])),
            float(np.sum(reference * ir[CH_Z][window])),
        ]
    )
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return None, (detected - expected) / SR * 1000.0
    return vector / norm, (detected - expected) / SR * 1000.0


def dry_signal(samples: int, seed: int) -> np.ndarray:
    """Band-limited noise bursts.

    Synthesised rather than taken from the call library on purpose: this is a
    localisation probe, and bursts with sharp onsets are what makes a moving
    direction audible. It also keeps the artifact free of any recording's
    licence.
    """

    rng = np.random.default_rng(seed)
    signal = np.zeros(samples, dtype=np.float64)
    burst = int(0.05 * SR)
    period = int(0.20 * SR)
    envelope = np.hanning(burst)
    for start in range(0, samples - burst, period):
        noise = rng.normal(0.0, 1.0, burst)
        # a cheap bandpass: difference then smooth, keeps energy mid-band
        noise = np.convolve(np.diff(noise, prepend=0.0), np.ones(4) / 4.0, mode="same")
        signal[start : start + burst] += envelope * noise
    peak = float(np.max(np.abs(signal)))
    return signal / peak if peak > 0 else signal


def write_wave(path: Path, channels: np.ndarray, sample_rate: int) -> None:
    peak = float(np.max(np.abs(channels)))
    scaled = channels / peak * 0.97 if peak > 0 else channels
    data = (scaled.T * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels.shape[0])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(data.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--motion-case", default="source1_moving_source2_static")
    parser.add_argument("--slot", default="source1")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--listener-height-m", type=float, default=1.5)
    parser.add_argument("--listener-minimum-range-m", type=float, default=2.0)
    parser.add_argument("--listener-maximum-range-m", type=float, default=6.0)
    parser.add_argument(
        "--listener-attempts",
        type=int,
        default=12,
        help=(
            "candidates to audition at the route midpoint. A candidate that "
            "fails there is behind a wall and would fail every frame"
        ),
    )
    parser.add_argument(
        "--skip-listener-audition",
        action="store_true",
        help="take the first candidate in the range band, occluded or not",
    )
    parser.add_argument(
        "--listener",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help=(
            "pin the listener to this world position instead of auditioning "
            "for one. Required for any A/B over rendering settings: the "
            "audition renders, so changing the acoustics changes which "
            "candidate passes, and the two runs then differ in listener as "
            "well as in the thing under test"
        ),
    )
    parser.add_argument(
        "--direct-window-ms",
        type=float,
        default=0.25,
        help=(
            "analysis window after the onset. 0.25 ms is the only length that "
            "held across four routes once real materials were in play; sweep it "
            "rather than trusting the default"
        ),
    )
    parser.add_argument(
        "--window-sweep",
        action="store_true",
        help=(
            "re-measure every rendered response at a range of windows and print "
            "the table. Costs nothing extra: the responses are already in hand"
        ),
    )
    parser.add_argument("--tolerance-deg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--dataset-config",
        help=(
            "vertex-colour semantic scene dataset config; giving it turns on "
            "acoustic materials. Without it every surface is the default "
            "material, which over-estimates reverberation about 2.5x"
        ),
    )
    parser.add_argument("--scene-id", help="e.g. 00800-TEEsavR23oF, with --dataset-config")
    parser.add_argument(
        "--materials-json",
        default="/data/jzy/code/sound-spaces/data/mp3d_material_config.json",
    )
    args = parser.parse_args()

    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    episodes = [e for e in bank["episodes"] if e["motion_case"] == args.motion_case]
    if not episodes:
        raise SystemExit(f"no {args.motion_case} episodes in {args.bank}")
    episode = episodes[args.episode_index]

    floor_path = np.asarray(episode["source_center_paths_m"][args.slot], dtype=float)
    centre_height = float(bank["source_center_heights_m"][args.slot])
    emitters = floor_path.copy()
    emitters[:, 1] += centre_height

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sim = make_sim(
        bank["scene"],
        bank.get("navmesh"),
        dataset_config=args.dataset_config,
        scene_id=args.scene_id,
        materials_json=args.materials_json,
    )
    materials_on = bool(args.dataset_config and args.scene_id)
    print(f"acoustic materials: {'on' if materials_on else 'off (default material)'}")

    midpoint = emitters[len(emitters) // 2]
    rng = np.random.default_rng(args.seed)
    agent = sim.get_agent(0)
    sensor = agent._sensors["audio_sensor"]

    def place(position):
        state = agent.get_state()
        state.position = np.asarray(position, dtype=np.float32)
        state.rotation = np.quaternion(1.0, 0.0, 0.0, 0.0)
        state.sensor_states = {}
        agent.set_state(state, True)

    def audition(candidate):
        """Error at the route midpoint, which is what decides this candidate."""

        place(candidate)
        sensor.setAudioSourceTransform(midpoint.astype(np.float32))
        ir = np.asarray(
            sim.get_sensor_observations()["audio_sensor"], dtype=np.float64
        )
        geometric = midpoint - candidate
        distance = float(np.linalg.norm(geometric))
        measured, _ = direct_arrival_direction(
            ir, distance, args.direct_window_ms
        )
        if measured is None:
            return None
        cosine = float(
            np.clip(float(np.dot(measured, geometric / distance)), -1.0, 1.0)
        )
        return float(np.degrees(np.arccos(cosine)))

    listener = None
    auditions = []
    if args.listener:
        listener = np.asarray(args.listener, dtype=np.float64)
        print(f"listener pinned to {np.round(listener, 3).tolist()}")
        error = audition(listener)
        auditions.append(
            {
                "position_m": [round(float(v), 4) for v in listener],
                "range_to_midpoint_m": round(
                    float(np.linalg.norm(listener - midpoint)), 3
                ),
                "midpoint_error_deg": None if error is None else round(error, 3),
                "pinned": True,
            }
        )
        print(
            f"  pinned listener midpoint error "
            f"{'none' if error is None else f'{error:.2f}'} deg"
        )
    for _ in range(0 if listener is not None else 4000):
        candidate = np.asarray(
            sim.pathfinder.get_random_navigable_point(), dtype=np.float64
        )
        if abs(candidate[1] - float(bank["floor_height_m"])) > 0.5:
            continue
        candidate[1] = float(bank["floor_height_m"]) + args.listener_height_m
        span = float(np.linalg.norm(candidate - midpoint))
        if not (
            args.listener_minimum_range_m <= span <= args.listener_maximum_range_m
        ):
            continue
        if args.skip_listener_audition:
            listener = candidate
            break
        error = audition(candidate)
        auditions.append(
            {
                "position_m": [round(float(v), 4) for v in candidate],
                "range_to_midpoint_m": round(span, 3),
                "midpoint_error_deg": None if error is None else round(error, 3),
            }
        )
        print(
            f"  audition {len(auditions):>2}  range {span:5.2f} m  "
            f"midpoint error {'none' if error is None else f'{error:6.2f}'} deg",
            flush=True,
        )
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
    place(listener)

    frames = list(range(0, len(emitters), args.frame_stride))
    responses = []
    per_frame = []
    for index in frames:
        emitter = emitters[index]
        sensor.setAudioSourceTransform(emitter.astype(np.float32))
        ir = np.asarray(sim.get_sensor_observations()["audio_sensor"], dtype=np.float64)
        responses.append(ir)
        print(f"  frame {index:>3}/{len(emitters) - 1}  ir {ir.shape[1]:>6} samples", flush=True)
        geometric = emitter - listener
        distance = float(np.linalg.norm(geometric))
        geometric = geometric / distance
        measured, onset_gap = direct_arrival_direction(
            ir, distance, args.direct_window_ms
        )
        if measured is None:
            per_frame.append(
                {"frame": index, "range_m": round(distance, 3), "error_deg": None}
            )
            continue
        cosine = float(np.clip(float(np.dot(measured, geometric)), -1.0, 1.0))
        per_frame.append(
            {
                "frame": index,
                "range_m": round(distance, 3),
                "error_deg": round(float(np.degrees(np.arccos(cosine))), 3),
                "onset_minus_direct_ms": round(onset_gap, 3),
            }
        )

    errors = np.array(
        [f["error_deg"] for f in per_frame if f["error_deg"] is not None], dtype=float
    )
    within = int(np.count_nonzero(errors <= args.tolerance_deg))

    # Time-varying convolution. Frame k's block of dry signal meets frame k's
    # response; overlap-add carries each response's tail into the next block, so
    # the reverberation does not restart every frame.
    hop = int(round(SR / float(bank["frame_rate_hz"]))) * args.frame_stride
    total = hop * len(frames)
    dry = dry_signal(total, args.seed)
    tail = max(ir.shape[1] for ir in responses)
    wet = np.zeros((4, total + tail), dtype=np.float64)
    for position, ir in enumerate(responses):
        block = dry[position * hop : (position + 1) * hop]
        if not block.size:
            continue
        for channel in range(4):
            piece = np.convolve(block, ir[channel])
            wet[channel, position * hop : position * hop + piece.size] += piece

    foa = args.output_dir / "moving_source.foa.wav"
    write_wave(foa, wet, SR)
    # +X is to the listener's right with an identity agent rotation, so a
    # virtual cardioid pair on that axis makes the movement audible directly.
    stereo = np.stack(
        (wet[0] - wet[CH_X], wet[0] + wet[CH_X]), axis=0
    )
    write_wave(args.output_dir / "moving_source.stereo.wav", stereo, SR)
    # Each render comes back with its own length - RLR returns as much tail as
    # the propagation produced - so they are padded to a common length rather
    # than stacked as-is, which raises.
    stacked = np.zeros((len(responses), 4, tail), dtype=np.float32)
    for position, ir in enumerate(responses):
        stacked[position, :, : ir.shape[1]] = ir
    np.save(args.output_dir / "moving_source_foa_frames.npy", stacked)

    sweep = None
    if args.window_sweep:
        windows = (0.25, 0.5, 1.0, 2.0, 4.0, 12.0)
        sweep = {}
        print("\nwindow sweep over the responses already rendered")
        print(f"{'range':>6} " + " ".join(f"{w:>7.2f}ms" for w in windows))
        table = []
        for position, index in enumerate(frames):
            emitter = emitters[index]
            geometric = emitter - listener
            distance = float(np.linalg.norm(geometric))
            geometric = geometric / distance
            row = []
            for window in windows:
                measured, _ = direct_arrival_direction(
                    responses[position], distance, window
                )
                if measured is None:
                    row.append(float("nan"))
                    continue
                cosine = float(np.clip(float(np.dot(measured, geometric)), -1.0, 1.0))
                row.append(float(np.degrees(np.arccos(cosine))))
            table.append(row)
            print(f"{distance:6.2f} " + " ".join(f"{v:9.2f}" for v in row))
        table = np.asarray(table)
        for column, window in enumerate(windows):
            values = table[:, column]
            sweep[f"{window}ms"] = {
                "median_deg": round(float(np.median(values)), 3),
                "maximum_deg": round(float(values.max()), 3),
                "frames_within_tolerance": int(
                    np.count_nonzero(values <= args.tolerance_deg)
                ),
            }
            print(
                f"  window {window:5.2f} ms   median {np.median(values):6.2f}  "
                f"max {values.max():6.2f}  within {args.tolerance_deg:g} deg "
                f"{int((values <= args.tolerance_deg).sum())}/{len(table)}"
            )

    report = {
        "schema": "avengine_moving_source_foa_v1",
        "scene": bank["scene"],
        "bank": str(args.bank),
        "episode_id": episode["episode_id"],
        "motion_case": episode["motion_case"],
        "slot": args.slot,
        "renderer": (
            "SoundSpaces habitat audio sensor, 4-channel ambisonics; the "
            "AVEngine runtime prefix does not wire the upstream audio sensor"
        ),
        "listener_m": [round(float(v), 4) for v in listener],
        "listener_auditions": auditions,
        "listener_height_above_floor_m": args.listener_height_m,
        "source_center_height_m": centre_height,
        "frames_rendered": len(frames),
        "frame_stride": args.frame_stride,
        "seconds": round(total / SR, 3),
        "direct_window_ms": args.direct_window_ms,
        "tolerance_deg": args.tolerance_deg,
        "direction_error_deg": {
            "median": round(float(np.median(errors)), 3),
            "p90": round(float(np.percentile(errors, 90)), 3),
            "maximum": round(float(errors.max()), 3),
        },
        "acoustic_materials": materials_on,
        "reverberation": {
            "t20_ms_median": None,
            "note": "measured per frame from the rendered responses below",
        },
        "frames_within_tolerance": within,
        "frames_measured": int(errors.size),
        "range_m": {
            "minimum": round(float(min(f["range_m"] for f in per_frame)), 3),
            "maximum": round(float(max(f["range_m"] for f in per_frame)), 3),
        },
        "per_frame": per_frame,
    }
    if sweep is not None:
        report["window_sweep"] = sweep
    (args.output_dir / "moving_source_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    # What reverberation these routes are actually rendered into, which is the
    # number the material switch moves.
    def t20(w):
        peak = int(np.argmax(np.abs(w)))
        tail = w[peak:].astype(float)
        if tail.size < 16:
            return float("nan")
        energy = np.cumsum(tail[::-1] ** 2)[::-1]
        curve = 10.0 * np.log10(np.maximum(energy / energy[0], 1e-20))
        a = np.flatnonzero(curve <= -5.0)
        b = np.flatnonzero(curve <= -25.0)
        return float("nan") if not a.size or not b.size else float(
            (b[0] - a[0]) / SR * 1000.0
        )

    decays = np.array([t20(ir[0]) for ir in responses], dtype=float)
    decays = decays[np.isfinite(decays)]
    if decays.size:
        report["reverberation"] = {
            "t20_ms_median": round(float(np.median(decays)), 2),
            "t20_ms_minimum": round(float(decays.min()), 2),
            "t20_ms_maximum": round(float(decays.max()), 2),
            "materials": materials_on,
        }
        print(
            f"T20 along the route  median {np.median(decays):.1f} ms  "
            f"({decays.min():.1f}-{decays.max():.1f})"
        )

    print(f"episode {episode['episode_id']}  slot {args.slot}")
    print(f"listener {np.round(listener, 3).tolist()}")
    print(
        f"range {report['range_m']['minimum']:.2f}-{report['range_m']['maximum']:.2f} m"
        f"  frames {len(frames)}  {report['seconds']:.2f} s"
    )
    print(
        f"DoA error  median {report['direction_error_deg']['median']:.2f}  "
        f"p90 {report['direction_error_deg']['p90']:.2f}  "
        f"max {report['direction_error_deg']['maximum']:.2f} deg"
    )
    print(f"within {args.tolerance_deg:g} deg: {within}/{errors.size} frames")
    print(f"wrote {args.output_dir}")
    sim.close()
    return 0 if within == errors.size else 1


if __name__ == "__main__":
    raise SystemExit(main())
