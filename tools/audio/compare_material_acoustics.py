"""Does turning HM3D semantics into acoustic materials change the sound?

Every tool in tools/audio has been rendering with enableMaterials off, which
gives one default material everywhere. JAEGER's paper renders "with
material-dependent acoustics" over the semantically annotated HM3D subset. This
measures what that switch is actually worth, on the same scene and the same
source/listener pairs.

Three things have to line up before materials do anything, and skipping any one
of them fails quietly rather than loudly:

  * the scene has to be opened through the annotated scene dataset config, not
    as a bare glb path. A raw path has no semantic association, and the audio
    sensor then logs "Semantic scene does not exist" and uses the default.
  * load_semantic_mesh has to be on.
  * the material database has to be handed to the sensor explicitly with
    setAudioMaterialsJSON. enableMaterials alone only says "look for one".

The database matching is by semantic label, and its vocabulary is worth
checking rather than assuming: mp3d_material_config.json carries 24 materials
under 64 labels, and those labels were written for MP3D. Notably neither floor
nor ceiling appears among them, and the carpet material carries no labels at
all, so those surfaces fall back to the default however good the annotations
are. The report counts how many of the scene's own categories matched.

enableMaterials cannot be changed after the mesh loads, so each setting gets its
own Simulator rather than one being reconfigured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import quaternion  # noqa: F401
import habitat_sim
from habitat_sim.sensor import AudioSensorSpec, RLRAudioPropagationChannelLayoutType

SR = 16000
CH_X, CH_Y, CH_Z = 3, 1, 2


def build(scene_id, dataset_config, materials_json, use_materials):
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene_id
    if dataset_config:
        backend.scene_dataset_config_file = str(dataset_config)
    backend.load_semantic_mesh = bool(use_materials)
    backend.enable_physics = False
    sim = habitat_sim.Simulator(
        habitat_sim.Configuration(backend, [habitat_sim.agent.AgentConfiguration()])
    )
    spec = AudioSensorSpec()
    spec.uuid = "audio_sensor"
    spec.enableMaterials = bool(use_materials)
    spec.channelLayout.type = RLRAudioPropagationChannelLayoutType.Ambisonics
    spec.channelLayout.channelCount = 4
    spec.position = [0.0, 0.0, 0.0]
    spec.acousticsConfig.sampleRate = SR
    spec.acousticsConfig.directSHOrder = 1
    spec.acousticsConfig.indirectSHOrder = 1
    spec.acousticsConfig.indirect = True
    sim.add_sensor(spec)
    sensor = sim.get_agent(0)._sensors["audio_sensor"]
    if use_materials and materials_json:
        sensor.setAudioMaterialsJSON(str(materials_json))
    return sim, sensor


def onset(w):
    return int(np.argmax(np.abs(w) > 0.15 * np.max(np.abs(w)) + 1e-12))


def t20_ms(w):
    peak = int(np.argmax(np.abs(w)))
    tail = w[peak:].astype(np.float64)
    if tail.size < 16:
        return float("nan")
    energy = np.cumsum(tail[::-1] ** 2)[::-1]
    curve = 10.0 * np.log10(np.maximum(energy / energy[0], 1e-20))
    start = np.flatnonzero(curve <= -5.0)
    end = np.flatnonzero(curve <= -25.0)
    if not start.size or not end.size:
        return float("nan")
    return float((end[0] - start[0]) / SR * 1000.0)


def direct_to_reverb_db(w):
    start = onset(w)
    n = int(0.0025 * SR)
    direct = float(np.sum(w[start : start + n] ** 2))
    late = float(np.sum(w[start + n :] ** 2))
    if direct <= 0 or late <= 0:
        return float("nan")
    return 10.0 * np.log10(direct / late)


def doa_error(ir, listener, source):
    w = ir[0]
    start = onset(w)
    n = max(int(0.001 * SR), 1)
    sl = slice(start, start + n)
    ref = w[sl]
    v = np.array([float(np.sum(ref * ir[c][sl])) for c in (CH_X, CH_Y, CH_Z)])
    if np.linalg.norm(v) <= 0:
        return float("nan")
    v = v / np.linalg.norm(v)
    g = np.asarray(source) - np.asarray(listener)
    g = g / np.linalg.norm(g)
    return float(np.degrees(np.arccos(np.clip(float(np.dot(v, g)), -1.0, 1.0))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True, help="e.g. 00800-TEEsavR23oF")
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--materials-json", required=True, type=Path)
    parser.add_argument("--jaeger-scene-dir", type=Path, default=None)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    # Source/listener pairs come from JAEGER's own tasks when available, so the
    # comparison sits on geometry that is known to be legal rather than on
    # points this script invented.
    pairs = []
    if args.jaeger_scene_dir:
        for task in sorted(args.jaeger_scene_dir.glob("task1_*"))[: args.pairs]:
            meta = json.loads((task / "metadata.json").read_text())
            pairs.append(
                (
                    np.asarray(meta["agent_pos"], dtype=np.float64),
                    np.asarray(meta["agent_rot_quat"], dtype=np.float64),
                    np.asarray(
                        meta["audio_source"]["position_world"], dtype=np.float64
                    ),
                    task.name,
                )
            )
    if not pairs:
        raise SystemExit("no source/listener pairs; pass --jaeger-scene-dir")

    rows = []
    coverage = {}
    for use_materials in (False, True):
        sim, sensor = build(
            args.scene_id, args.dataset_config, args.materials_json, use_materials
        )
        if use_materials:
            labels = set()
            database = json.loads(args.materials_json.read_text())
            for material in database["materials"]:
                labels.update(material.get("labels", []))
            scene = sim.semantic_scene
            categories = sorted(
                {
                    obj.category.name()
                    for obj in getattr(scene, "objects", []) or []
                    if obj is not None and obj.category is not None
                }
            )
            matched = [c for c in categories if c in labels]
            coverage = {
                "scene_categories": len(categories),
                "database_labels": len(labels),
                "categories_matched": len(matched),
                "matched_examples": matched[:12],
                "unmatched_examples": [c for c in categories if c not in labels][:12],
            }
            print(
                f"semantic categories in scene {len(categories)}, "
                f"matched to a material {len(matched)}"
            )
            print(f"  matched: {', '.join(matched[:10])}")
            print(f"  unmatched: {', '.join(coverage['unmatched_examples'][:10])}")

        agent = sim.get_agent(0)
        for listener, rotation, source, name in pairs:
            state = agent.get_state()
            state.position = listener.astype(np.float32)
            state.rotation = np.quaternion(*rotation)
            state.sensor_states = {}
            agent.set_state(state, True)
            sensor.setAudioSourceTransform(source.astype(np.float32))
            ir = np.asarray(
                sim.get_sensor_observations()["audio_sensor"], dtype=np.float64
            )
            rows.append(
                {
                    "task": name,
                    "materials": use_materials,
                    "range_m": round(float(np.linalg.norm(source - listener)), 3),
                    "ir_samples": int(ir.shape[1]),
                    "t20_ms": round(t20_ms(ir[0]), 2),
                    "direct_to_reverb_db": round(direct_to_reverb_db(ir[0]), 2),
                    "doa_error_deg": round(doa_error(ir, listener, source), 3),
                }
            )
        sim.close()

    print(f"\n{'task':<14} {'range':>6} {'materials':>10} {'samples':>8} "
          f"{'T20_ms':>7} {'D/R_dB':>7} {'DoA':>6}")
    for row in rows:
        print(
            f"{row['task']:<14} {row['range_m']:6.2f} {str(row['materials']):>10} "
            f"{row['ir_samples']:8d} {row['t20_ms']:7.1f} "
            f"{row['direct_to_reverb_db']:7.2f} {row['doa_error_deg']:6.2f}"
        )

    off = [r for r in rows if not r["materials"]]
    on = [r for r in rows if r["materials"]]
    if off and on:
        print(
            f"\nT20 median  materials off {np.median([r['t20_ms'] for r in off]):.1f} ms"
            f"   on {np.median([r['t20_ms'] for r in on]):.1f} ms"
        )
        print(
            f"D/R median  materials off "
            f"{np.median([r['direct_to_reverb_db'] for r in off]):.2f} dB"
            f"   on {np.median([r['direct_to_reverb_db'] for r in on]):.2f} dB"
        )
    if args.report:
        args.report.write_text(
            json.dumps(
                {
                    "schema": "avengine_material_acoustics_comparison_v1",
                    "scene_id": args.scene_id,
                    "materials_json": str(args.materials_json),
                    "label_coverage": coverage,
                    "rows": rows,
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
