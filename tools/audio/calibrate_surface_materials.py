"""Ask which material the room's dominant surfaces should point at.

Wall, ceiling and floor are about 59 percent of the annotated area between them
in HM3D, so whichever materials they name decide the reverberation and every
other label is a rounding correction. This re-points one label at a time and
measures what the room then does.

The reference is a literature range, not a measurement of these houses: a
furnished domestic room sits around 0.3 to 0.6 s of mid-band T60. There is no
measured impulse response for these specific scans, so this cannot claim to
reproduce them - it can only say which assignment lands a scanned house inside
the range real houses occupy, and which puts it outside.

T60 is extrapolated from T30 and from T20, and both are reported alongside the
straightness of the decay they came from. An extrapolation from a decay that is
not straight is arithmetic rather than a reverberation time, so the correlation
is printed and a poor one should be read as "this number means nothing" instead
of as a short room.

Reading the material data needs care and cost me a wrong conclusion already.
absorption is a flat [frequency, alpha, frequency, alpha, ...] list, so
absorption[1] is the coefficient at the lowest frequency in the table, usually
125 Hz. Taking it for the mid-band value made Gypsum Board look far too
absorptive at 0.29 when its mid-band is 0.053, and made the floor's Carpet look
like a hard reflector at 0.010 when its mid-band is 0.25.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import quaternion  # noqa: F401
import habitat_sim
from habitat_sim.sensor import AudioSensorSpec, RLRAudioPropagationChannelLayoutType


def mid_band_alpha(material):
    flat = material["absorption"]
    pairs = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    mid = [alpha for freq, alpha in pairs if 400.0 <= freq <= 2000.0]
    return sum(mid) / len(mid) if mid else float("nan")


def repoint(base, label, material_name):
    """Move one label to another material, leaving every coefficient alone."""

    config = json.loads(json.dumps(base))
    names = [m["name"] for m in config["materials"]]
    if material_name not in names:
        raise SystemExit(f"no material called {material_name!r}")
    for material in config["materials"]:
        material["labels"] = [l for l in material.get("labels", []) if l != label]
    for material in config["materials"]:
        if material["name"] == material_name:
            material.setdefault("labels", []).append(label)
    return config


def decay(w, sample_rate, drop_db):
    """Schroeder decay time over -5 dB to -(5+drop), with its straightness."""

    peak = int(np.argmax(np.abs(w)))
    tail = np.asarray(w[peak:], dtype=float)
    if tail.size < 64:
        return float("nan"), float("nan")
    energy = np.cumsum(tail[::-1] ** 2)[::-1]
    curve = 10.0 * np.log10(np.maximum(energy / energy[0], 1e-20))
    inside = np.flatnonzero((curve <= -5.0) & (curve >= -(5.0 + drop_db)))
    if inside.size < 16:
        return float("nan"), float("nan")
    x = inside.astype(float) / sample_rate
    y = curve[inside]
    slope, intercept = np.polyfit(x, y, 1)
    if slope >= 0:
        return float("nan"), float("nan")
    fit = slope * x + intercept
    residual = float(np.sum((y - fit) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    straightness = 1.0 - residual / total if total > 0 else float("nan")
    return float(-60.0 / slope), straightness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-materials", required=True, type=Path)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--scene-id", required=True, action="append")
    parser.add_argument("--jaeger-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    base = json.loads(args.base_materials.read_text(encoding="utf-8"))
    by_name = {m["name"]: m for m in base["materials"]}
    current = next(
        (m["name"] for m in base["materials"]
         if args.label in m.get("labels", [])),
        None,
    )
    print(f"label {args.label!r} currently points at {current!r}"
          f" (mid-band alpha "
          f"{mid_band_alpha(by_name[current]):.3f})" if current else
          f"label {args.label!r} is unmapped")

    rows = []
    with tempfile.TemporaryDirectory() as work:
        for candidate in args.candidate:
            config = repoint(base, args.label, candidate)
            path = Path(work) / f"{candidate.replace(' ', '_').replace(',', '')}.json"
            path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
            alpha = mid_band_alpha(by_name[candidate])
            per_scene = []
            for scene_id in args.scene_id:
                backend = habitat_sim.SimulatorConfiguration()
                backend.scene_id = scene_id
                backend.scene_dataset_config_file = args.dataset_config
                backend.load_semantic_mesh = True
                backend.enable_physics = False
                sim = habitat_sim.Simulator(
                    habitat_sim.Configuration(
                        backend, [habitat_sim.agent.AgentConfiguration()]
                    )
                )
                spec = AudioSensorSpec()
                spec.uuid = "audio_sensor"
                spec.enableMaterials = True
                spec.channelLayout.type = (
                    RLRAudioPropagationChannelLayoutType.Ambisonics
                )
                spec.channelLayout.channelCount = 4
                spec.position = [0.0, 0.0, 0.0]
                spec.acousticsConfig.sampleRate = args.sample_rate
                spec.acousticsConfig.indirect = True
                sim.add_sensor(spec)
                sensor = sim.get_agent(0)._sensors["audio_sensor"]
                sensor.setAudioMaterialsJSON(str(path))
                agent = sim.get_agent(0)
                scene_dir = args.jaeger_root / scene_id
                for task in sorted(scene_dir.glob("task1_*"))[: args.pairs]:
                    meta = json.loads((task / "metadata.json").read_text())
                    state = agent.get_state()
                    state.position = np.asarray(meta["agent_pos"], np.float32)
                    state.rotation = np.quaternion(*meta["agent_rot_quat"])
                    state.sensor_states = {}
                    agent.set_state(state, True)
                    sensor.setAudioSourceTransform(
                        np.asarray(
                            meta["audio_source"]["position_world"], np.float32
                        )
                    )
                    ir = np.asarray(
                        sim.get_sensor_observations()["audio_sensor"], dtype=float
                    )
                    t60_30, straight30 = decay(ir[0], args.sample_rate, 30.0)
                    t60_20, straight20 = decay(ir[0], args.sample_rate, 20.0)
                    per_scene.append(
                        {
                            "scene": scene_id,
                            "task": task.name,
                            "t60_from_t30_s": None if math.isnan(t60_30)
                            else round(t60_30, 3),
                            "t60_from_t20_s": None if math.isnan(t60_20)
                            else round(t60_20, 3),
                            "straightness_t30": None if math.isnan(straight30)
                            else round(straight30, 4),
                        }
                    )
                sim.close()
            usable = [r["t60_from_t30_s"] for r in per_scene
                      if r["t60_from_t30_s"] is not None]
            straight = [r["straightness_t30"] for r in per_scene
                        if r["straightness_t30"] is not None]
            row = {
                "candidate": candidate,
                "mid_band_alpha": round(alpha, 3),
                "t60_median_s": round(float(np.median(usable)), 3) if usable else None,
                "t60_minimum_s": round(float(min(usable)), 3) if usable else None,
                "t60_maximum_s": round(float(max(usable)), 3) if usable else None,
                "decay_straightness_median": round(float(np.median(straight)), 4)
                if straight else None,
                "measurements": per_scene,
            }
            rows.append(row)
            print(
                f"  {candidate:<26} alpha {alpha:5.3f}   "
                f"T60 {row['t60_median_s']} s "
                f"({row['t60_minimum_s']}-{row['t60_maximum_s']})   "
                f"decay straightness {row['decay_straightness_median']}"
            )

    print("\nfurnished domestic rooms sit around 0.3-0.6 s of mid-band T60; "
          "this is a literature range, not a measurement of these scans")
    for row in rows:
        if row["t60_median_s"] is None:
            verdict = "unmeasurable"
        elif 0.3 <= row["t60_median_s"] <= 0.6:
            verdict = "inside the residential range"
        elif row["t60_median_s"] < 0.3:
            verdict = "too dead for a house"
        else:
            verdict = "too live for a house"
        print(f"  {row['candidate']:<26} {row['t60_median_s']} s  {verdict}")

    if args.report:
        args.report.write_text(
            json.dumps(
                {
                    "schema": "avengine_surface_material_calibration_v1",
                    "label": args.label,
                    "currently": current,
                    "reference_range_s": [0.3, 0.6],
                    "reference_note": (
                        "literature range for furnished domestic rooms; no "
                        "measured impulse response exists for these scans"
                    ),
                    "candidates": rows,
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
