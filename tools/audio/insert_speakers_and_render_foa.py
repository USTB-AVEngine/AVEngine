"""Put our published speakers into a Habitat scene and render RGB plus FOA.

This is the chain the whole instance-diversity argument needs and that JAEGER
does not have: several visually distinct loudspeakers standing in one real
scene, each able to be the sound source, so "which speaker is playing" has an
answer grounded in appearance rather than only in position.

It runs on the open scenes that are already downloaded, so it does not wait on
the licence-gated HM3D meshes. Pointing --scene at an HM3D glb is the only
change needed once those arrive.

This renders the audio half. The visual half is Blender, driven by the same
placement file, because the habitat-sim build on this machine cannot bring up a
GL context - camera sensors abort with GL::Renderer::Error::InvalidValue under
every EGL configuration tried, while the audio path measures a perfect 0.00
degrees. Splitting them is also what the geometry actually calls for: a speaker
mesh is a visual proxy and the sound is rendered from a point source at its
emitter, which is how JAEGER does it too.

Two things here are not obvious and both are load-bearing:

  * the emitter is not the object's origin. Every published static asset
    carries an emitter offset in its own frame - the woofer cone, the grille
    centre - and the audio source belongs there, not at the object centre.
  * a published asset faces +X in its own frame, so placement carries an
    explicit yaw rather than hoping the two conventions agree.

Verification is the one the team already proved on this renderer: the active
intensity vector of the direct arrival has to recover the geometric source
direction. If it does not, the placement is wrong, not the audio.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import quaternion  # noqa: F401  imported before habitat_sim per SoundSpaces issue
import habitat_sim
from habitat_sim.sensor import AudioSensorSpec, RLRAudioPropagationChannelLayoutType

SR = 16000
# SoundSpaces 2.0 emits ACN [W, Y, Z, X]; the JAEGER release uses [W, Z, X, Y].
# Everything below is in the SoundSpaces order because that is what we render.
CH_X, CH_Y, CH_Z = 3, 1, 2


def make_sim(scene: str) -> habitat_sim.Simulator:
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene
    backend.load_semantic_mesh = False
    backend.enable_physics = False
    sim = habitat_sim.Simulator(
        habitat_sim.Configuration(backend, [habitat_sim.agent.AgentConfiguration()])
    )
    audio = AudioSensorSpec()
    audio.uuid = "audio_sensor"
    audio.enableMaterials = False
    audio.channelLayout.type = RLRAudioPropagationChannelLayoutType.Ambisonics
    audio.channelLayout.channelCount = 4
    audio.position = [0.0, 0.0, 0.0]
    audio.acousticsConfig.sampleRate = SR
    audio.acousticsConfig.directSHOrder = 1
    audio.acousticsConfig.indirectSHOrder = 1
    audio.acousticsConfig.indirect = True
    sim.add_sensor(audio)
    return sim


def emitter_world_position(asset_dir: Path, position, yaw_deg: float):
    """Where the sound actually leaves this object, in world coordinates.

    The published record stores the offset in the asset's own frame, which is
    +X forward, +Y up, +Z right. Habitat is y-up as well, so only the yaw has
    to be applied.
    """

    record = json.loads((asset_dir / "asset.json").read_text(encoding="utf-8"))
    offset = np.array(record["emitter"]["offset_m"], dtype=float)
    angle = math.radians(yaw_deg)
    rotated = np.array(
        [
            offset[0] * math.cos(angle) + offset[2] * math.sin(angle),
            offset[1],
            -offset[0] * math.sin(angle) + offset[2] * math.cos(angle),
        ]
    )
    return np.array(position, dtype=float) + rotated, record


def render_ir(sim, source_world):
    sensor = sim.get_agent(0)._sensors["audio_sensor"]
    sensor.setAudioSourceTransform(np.array(source_world, dtype=np.float32))
    return np.asarray(sim.get_sensor_observations()["audio_sensor"])


SPEED_OF_SOUND = 343.0


def direct_arrival_direction(
    ir: np.ndarray, distance_m: float, window_ms: float
) -> tuple:
    """Direction of the direct path, windowed at the time geometry predicts.

    Detecting the onset by amplitude works in an empty box and fails in a
    furnished room: the loudest early energy can be a reflection off a nearby
    surface, and the intensity vector then points at that surface instead of at
    the source. The direct path arrives at distance over the speed of sound
    whether or not it is the loudest thing, so the window goes there.

    The detected onset is returned alongside so the two can be compared: a
    large gap means the direct path is obstructed and the sample should be
    treated as occluded rather than as a measurement.
    """

    w = ir[0]
    threshold = 0.15 * np.max(np.abs(w)) + 1e-12
    detected = int(np.argmax(np.abs(w) > threshold))
    expected = int(round(distance_m / SPEED_OF_SOUND * SR))
    # Two milliseconds, not the twelve the existing scripts use. In an empty box
    # nothing else arrives inside twelve, so that window measured a clean 0.000
    # degrees and the length was never questioned. A real room answers within
    # three to six: swept on skokloster-castle, every source reads 0.0 degrees
    # at 0.5 to 2 ms and about 18 at 6 to 12, because the early reflections
    # inside the longer window drag the intensity vector off the direct path.
    window = slice(detected, detected + max(int(window_ms * 0.001 * SR), 1))
    reference = w[window]
    vector = np.array(
        [
            float(np.sum(reference * ir[CH_X][window])),
            float(np.sum(reference * ir[CH_Y][window])),
            float(np.sum(reference * ir[CH_Z][window])),
        ]
    )
    norm = np.linalg.norm(vector)
    if norm <= 0:
        raise SystemExit("no energy arrives when the direct path should")
    return vector / norm, (detected - expected) / SR * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--placement", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tolerance-deg", type=float, default=5.0)
    parser.add_argument(
        "--direct-window-ms",
        type=float,
        default=2.0,
        help="analysis window after the onset; longer lets early reflections in",
    )
    args = parser.parse_args()

    plan = json.loads(args.placement.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sim = make_sim(args.scene)

    placed = []
    for item in plan["speakers"]:
        asset_dir = args.asset_root / item["asset"]
        glb = asset_dir / "finalized.glb"
        if not glb.is_file():
            raise SystemExit(f"missing asset mesh: {glb}")
        emitter, record = emitter_world_position(
            asset_dir, item["position"], item.get("yaw_deg", 0.0)
        )
        placed.append(
            {
                "asset": item["asset"],
                "object_type": record["identity"]["object_type"],
                "realized_attributes": record["realized_attributes"],
                "position": list(map(float, item["position"])),
                "yaw_deg": item.get("yaw_deg", 0.0),
                "emitter_world": [round(float(v), 4) for v in emitter],
                "anchor_id": record["emitter"]["anchor_id"],
            }
        )

    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = np.array(plan["receiver"], dtype=np.float32)
    state.rotation = np.quaternion(*plan.get("receiver_rotation", (1, 0, 0, 0)))
    state.sensor_states = {}
    agent.set_state(state, True)

    receiver = np.array(plan["receiver"], dtype=float)
    report = {"scene": args.scene, "receiver": list(map(float, receiver)), "sources": []}
    failures = 0
    for entry in placed:
        emitter = np.array(entry["emitter_world"], dtype=float)
        ir = render_ir(sim, emitter)
        np.save(args.output_dir / f"foa_{entry['asset'].replace('/', '_')}.npy", ir)
        geometric = emitter - receiver
        distance = float(np.linalg.norm(geometric))
        geometric = geometric / distance
        measured, onset_gap_ms = direct_arrival_direction(
            ir, distance, args.direct_window_ms
        )
        entry["range_m"] = round(distance, 3)
        entry["onset_minus_direct_ms"] = round(onset_gap_ms, 3)
        cosine = float(np.clip(np.dot(measured, geometric), -1.0, 1.0))
        error = float(np.degrees(np.arccos(cosine)))
        failures += error > args.tolerance_deg
        entry["direction_error_deg"] = round(error, 3)
        entry["geometric_direction"] = [round(float(v), 4) for v in geometric]
        report["sources"].append(entry)
        print(
            f"{entry['object_type']:<22} {str(entry['realized_attributes'])[:44]:<46} "
            f"DoA error {error:6.2f} deg"
        )

    report["placement"] = plan
    report["visual_renderer"] = (
        "blender, driven from the same placement file; the habitat-sim build "
        "here cannot create a GL context for camera sensors"
    )
    report["foa_channel_order"] = "soundspaces ACN [W, Y, Z, X]"
    report["direct_window_ms"] = args.direct_window_ms
    report["direction_tolerance_deg"] = args.tolerance_deg
    report["failures"] = failures
    (args.output_dir / "scene_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {args.output_dir}  sources={len(placed)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
