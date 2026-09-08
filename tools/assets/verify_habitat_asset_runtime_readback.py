#!/usr/bin/env python3
"""Read back Habitat native root, contact-link and emitter transforms.

This is a research-candidate evidence producer. It applies the exact first
Idle and Walking baked poses to each package's articulated object and records
the native scene-node transforms used for the root, declared contact links and
mouth emitter. It does not mutate a registry or claim formal capture.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from avengine.assets.actions import read_baked_actions_npz
from avengine.assets.glb import load_glb
from avengine.assets.habitat import HabitatLinkJointBlock, bind_habitat_link_layout, build_habitat_asset_mapping_from_rebase_report
from avengine.contracts.json_io import load_json


def _record(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path.resolve()), "byte_size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _transform_matrix(value: Mapping[str, Any]) -> np.ndarray:
    translation = np.asarray(value["translation_m"], dtype=np.float64)
    x, y, z, w = (float(item) for item in value["rotation_xyzw"])
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
    matrix[:3, 3] = translation
    return matrix


def _native_point(node: Any, mn: Any) -> list[float]:
    matrix = np.asarray(node.absolute_transformation(), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise RuntimeError("native link transform is not finite 4x4")
    point = np.asarray(matrix[:3, 3], dtype=np.float64)
    if not np.all(np.isfinite(point)):
        raise RuntimeError("native link translation is not finite")
    return [float(value) for value in point]


def _review_module(repo: Path) -> Any:
    path = repo / "tools/assets/render_habitat_action_review.py"
    module_spec = importlib.util.spec_from_file_location("habitat_action_review", path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("cannot import Habitat review helpers")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def readback(package_root: Path, *, gpu_device_id: int) -> Path:
    package_root = package_root.resolve()
    manifest_path = package_root / "package/asset_manifest.json"
    review_root = package_root / "habitat_animation_review"
    config_path = review_root / "animal.ao_config.json"
    actions_path = package_root / "actions.npz"
    rebased_path = package_root / "rebased.glb"
    rebase_report_path = package_root / "rebase.json"
    output_path = package_root / "native_readback.json"
    if output_path.exists() or output_path.is_symlink():
        raise RuntimeError(f"refusing to replace readback output: {output_path}")
    required = (manifest_path, config_path, actions_path, rebased_path, rebase_report_path)
    if any(not path.is_file() or path.is_symlink() for path in required):
        raise RuntimeError(f"package is missing native-readback inputs: {package_root}")
    manifest = load_json(manifest_path)
    package_mapping = build_habitat_asset_mapping_from_rebase_report(
        load_glb(rebased_path), load_json(rebase_report_path)
    )
    actions = read_baked_actions_npz(actions_path)
    if tuple(manifest["skeleton"]["runtime_joint_order"]) != package_mapping.runtime_joint_order:
        raise RuntimeError("package skeleton and Habitat mapping runtime order differ")
    anchors = {
        item["anchor_id"]: item
        for item in manifest["anchors"]
        if isinstance(item, Mapping)
    }
    contact_order = tuple(manifest["contacts"]["contact_order"])
    if "muzzle" not in anchors:
        raise RuntimeError("package has no muzzle anchor")
    review = _review_module(Path(__file__).resolve().parents[2])
    configuration, qt, mn = review._make_configuration()
    import habitat_sim

    sim_cfg = configuration.sim_cfg
    sim_cfg.gpu_device_id = 0
    results: dict[str, Any] = {}
    with habitat_sim.Simulator(configuration) as simulator:
        loaded = simulator.metadata_mediator.ao_template_manager.load_configs(str(config_path))
        handle_prefix = config_path.stem.removesuffix(".ao_config")
        handles = simulator.metadata_mediator.ao_template_manager.get_template_handles(handle_prefix)
        if len(loaded) != 1 or len(handles) != 1:
            raise RuntimeError(f"expected one AO template, got {loaded} / {handles}")
        actor = simulator.get_articulated_object_manager().add_articulated_object_by_template_handle(handles[0])
        if actor is None:
            raise RuntimeError("Habitat failed to instantiate the articulated object")
        actor.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        actual_names = {actor.get_link_name(-1)} | {actor.get_link_name(link_id) for link_id in actor.get_link_ids()}
        if actual_names != set(package_mapping.joint_order):
            raise RuntimeError(
                "native link names differ from package mapping: "
                f"missing={set(package_mapping.joint_order) - actual_names}, "
                f"extra={actual_names - set(package_mapping.joint_order)}"
            )
        blocks = [
            HabitatLinkJointBlock(
                link_name=actor.get_link_name(link_id),
                joint_position_offset=int(actor.get_link_joint_pos_offset(link_id)),
                joint_position_count=int(actor.get_link_num_joint_pos(link_id)),
            )
            for link_id in actor.get_link_ids()
        ]
        binding = bind_habitat_link_layout(
            package_mapping.runtime_joint_order,
            blocks,
            joint_position_count=len(actor.joint_positions),
        )
        actor_from_skin_root = np.asarray(package_mapping.actor_from_skin_root, dtype=np.float64)
        review._apply_root_transform(actor, actor_from_skin_root, qt=qt, mn=mn)
        link_ids_by_name = {
            actor.get_link_name(link_id): int(link_id) for link_id in actor.get_link_ids()
        }
        link_ids_by_name[actor.get_link_name(-1)] = -1
        emitter = anchors["muzzle"]
        emitter_matrix = _transform_matrix(emitter["joint_from_anchor"])
        world_time_before = float(simulator.get_world_time())
        for action_id in ("idle", "walk"):
            clip = actions.action(action_id)
            pose = clip.rotations_xyzw[0]
            actor.joint_positions = np.asarray(binding.map_pose(pose), dtype=np.float64)
            root_node = actor.get_link_scene_node(-1)
            root_matrix = np.asarray(root_node.absolute_transformation(), dtype=np.float64)
            if root_matrix.shape != (4, 4) or not np.all(np.isfinite(root_matrix)):
                raise RuntimeError(f"{action_id} root transform is not finite 4x4")
            link_positions = {}
            for anchor_id in contact_order:
                anchor = anchors[anchor_id]
                link_id = link_ids_by_name.get(anchor["joint_id"])
                if link_id is None:
                    raise RuntimeError(f"contact anchor {anchor_id} joint is absent in native links")
                link_positions[anchor_id] = _native_point(actor.get_link_scene_node(link_id), mn)
            emitter_link_id = link_ids_by_name.get(emitter["joint_id"])
            if emitter_link_id is None:
                raise RuntimeError("muzzle joint is absent in native links")
            emitter_transform = np.asarray(
                actor.get_link_scene_node(emitter_link_id).absolute_transformation(),
                dtype=np.float64,
            )
            emitter_world = emitter_transform @ emitter_matrix
            if emitter_world.shape != (4, 4) or not np.all(np.isfinite(emitter_world)):
                raise RuntimeError(f"{action_id} emitter transform is not finite 4x4")
            results[action_id] = {
                "sample_index": 0,
                "root": {
                    "link_name": actor.get_link_name(-1),
                    "translation_m": [float(value) for value in root_matrix[:3, 3]],
                    "matrix": root_matrix.tolist(),
                },
                "contacts": {
                    "contact_order": list(contact_order),
                    "native_link_positions_m": link_positions,
                },
                "emitter": {
                    "anchor_id": "muzzle",
                    "joint_id": emitter["joint_id"],
                    "world_position_m": [float(value) for value in emitter_world[:3, 3]],
                    "joint_world_position_m": [float(value) for value in emitter_transform[:3, 3]],
                    "joint_from_anchor": emitter["joint_from_anchor"],
                },
            }
        world_time_after = float(simulator.get_world_time())
    value = {
        "schema": "avengine_habitat_asset_native_readback_v1",
        "status": "pass",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "asset_id": manifest["asset_id"],
        "package_manifest": _record(manifest_path),
        "native_runtime": {
            "gpu_device_id": gpu_device_id,
            "simulator_scene": "NONE",
            "world_time_before": world_time_before,
            "world_time_after": world_time_after,
            "world_time_unchanged": world_time_before == world_time_after,
            "runtime_joint_order": list(package_mapping.runtime_joint_order),
            "root_transform_formula": "world_from_skin_root = world_from_actor @ actor_from_skin_root",
            "readback_actions": results,
        },
        "notes": [
            "Native scene-node transforms were read after applying the exact baked Idle and Walking first samples.",
            "Emitter and contact values are native transform readbacks; visual/audio capture and formal qualification remain separate evidence boundaries.",
        ],
    }
    output_path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--gpu-device-id", type=int, default=3)
    args = parser.parse_args()
    result = readback(args.package_root, gpu_device_id=args.gpu_device_id)
    print(json.dumps({"status": "pass", "native_readback": str(result)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
