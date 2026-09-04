"""Fit the map from SoundSpaces FOA channels to Habitat world axes.

The constant in render_real.py - CH_X, CH_Y, CH_Z = 3, 1, 2 with no signs and
no axis swap - was fitted on sources placed along the world axes, and along an
axis the fit is degenerate: several permutations tie, so the search returns
whichever it saw first. Two of the sources in that script's own output are
labelled as 90 degrees of elevation while lying flat on the floor, which is the
same mistake showing up in the ground truth.

Ambisonics is x-forward, y-left, z-up; Habitat is y-up. The two cannot be the
identity, so this fits the signed permutation directly, using source directions
that are deliberately not axis aligned so nothing is degenerate, in the box
room where the geometry is unambiguous.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import quaternion  # noqa: F401
import habitat_sim
from habitat_sim.sensor import AudioSensorSpec, RLRAudioPropagationChannelLayoutType

SR = 16000


def make_sim(scene: Path):
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(scene)
    backend.load_semantic_mesh = False
    backend.enable_physics = False
    sim = habitat_sim.Simulator(
        habitat_sim.Configuration(backend, [habitat_sim.agent.AgentConfiguration()])
    )
    spec = AudioSensorSpec()
    spec.uuid = "audio_sensor"
    spec.enableMaterials = False
    spec.channelLayout.type = RLRAudioPropagationChannelLayoutType.Ambisonics
    spec.channelLayout.channelCount = 4
    spec.position = [0.0, 0.0, 0.0]
    spec.acousticsConfig.sampleRate = SR
    spec.acousticsConfig.directSHOrder = 1
    spec.acousticsConfig.indirectSHOrder = 1
    spec.acousticsConfig.indirect = True
    sim.add_sensor(spec)
    return sim


def intensity(ir):
    w = ir[0]
    threshold = 0.15 * np.max(np.abs(w)) + 1e-12
    onset = int(np.argmax(np.abs(w) > threshold))
    window = slice(onset, onset + max(int(0.012 * SR), 1))
    reference = w[window]
    return np.array([float(np.sum(reference * ir[c][window])) for c in (1, 2, 3)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", required=True, type=Path,
        help="explicit non-axis-aligned calibration room GLB",
    )
    args = parser.parse_args(argv)
    scene = args.scene.expanduser().resolve()
    if not scene.is_file():
        parser.error(f"--scene is not a file: {scene}")
    sim = make_sim(scene)
    agent = sim.get_agent(0)
    receiver = np.array([3.0, 1.5, 2.0], dtype=np.float32)
    state = agent.get_state()
    state.position = receiver
    state.rotation = np.quaternion(1, 0, 0, 0)
    state.sensor_states = {}
    agent.set_state(state, True)
    sensor = agent._sensors["audio_sensor"]

    # Deliberately generic directions: no zero components, no equal magnitudes.
    offsets = [
        (0.9, 0.5, 1.3), (-1.1, 0.4, 0.7), (0.6, -0.5, -1.2), (-0.8, -0.4, 1.1),
        (1.2, 0.7, -0.6), (-0.5, 0.9, -1.0), (1.0, -0.6, 0.5), (-1.3, -0.7, -0.4),
    ]
    measured, truth = [], []
    for offset in offsets:
        source = receiver + np.array(offset, dtype=np.float32)
        sensor.setAudioSourceTransform(source)
        ir = np.asarray(sim.get_sensor_observations()["audio_sensor"])
        if ir.shape[0] != 4 and ir.shape[-1] == 4:
            ir = ir.T
        vector = intensity(ir)
        norm = np.linalg.norm(vector)
        if norm <= 0:
            continue
        measured.append(vector / norm)
        direction = np.array(offset, dtype=float)
        truth.append(direction / np.linalg.norm(direction))
    measured = np.array(measured)
    truth = np.array(truth)
    print(f"fitting on {len(measured)} generic directions\n")

    best = None
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            mapped = np.stack(
                [signs[i] * measured[:, perm[i]] for i in range(3)], axis=1
            )
            mapped /= np.linalg.norm(mapped, axis=1, keepdims=True)
            angles = np.degrees(
                np.arccos(np.clip(np.sum(mapped * truth, axis=1), -1.0, 1.0))
            )
            worst = float(angles.max())
            if best is None or worst < best[0]:
                best = (worst, perm, signs, float(angles.mean()))
    worst, perm, signs, mean = best
    names = "xyz"
    print("best signed permutation, worst-case over all directions:")
    for i in range(3):
        sign = "+" if signs[i] > 0 else "-"
        print(f"  world {names[i]}  =  {sign}channel[{perm[i] + 1}]")
    print(f"\n  worst {worst:.3f} deg   mean {mean:.3f} deg")
    print(f"\n  as constants: CH_X, CH_Y, CH_Z = "
          f"{perm[0] + 1}, {perm[1] + 1}, {perm[2] + 1}   signs = {signs}")
    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
