from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from avengine.optional_backends.spear_replicacad import (
    COORDINATE_CONVENTION,
    ReplicaCADPlanError,
    build_replicacad_scene_plan,
    habitat_position_to_unreal_cm,
    habitat_quaternion_wxyz_to_unreal_xyzw,
    habitat_scale_to_unreal,
)


def _json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _asset(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return path


@pytest.fixture
def replica_scene(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "replica_cad"
    dataset = _json(
        root / "replicaCAD.scene_dataset_config.json",
        {
            "stages": {"paths": {".json": ["configs/stages"]}},
            "objects": {"paths": {".json": ["configs/objects"]}},
            "articulated_objects": {"paths": {".urdf": ["urdf/*/"]}},
        },
    )
    _asset(root / "stages/room.glb")
    _asset(root / "objects/chair.glb")
    _asset(root / "objects/table.glb")
    stage = _json(
        root / "configs/stages/room.stage_config.json",
        {"render_asset": "../../stages/room.glb", "up": [0, 1, 0]},
    )
    chair = _json(
        root / "configs/objects/chair.object_config.json",
        {"render_asset": "../../objects/chair.glb"},
    )
    table = _json(
        root / "configs/objects/table.object_config.json",
        {"render_asset": "../../objects/table.glb"},
    )

    cabinet_root = root / "urdf/cabinet"
    _asset(cabinet_root / "cabinet_body.glb")
    _asset(cabinet_root / "cabinet_door.glb")
    urdf = cabinet_root / "cabinet.urdf"
    urdf.write_text(
        """<?xml version="1.0"?>
<robot name="cabinet">
  <link name="root"/>
  <joint name="root_fixed" type="fixed">
    <parent link="root"/><child link="body"/>
  </joint>
  <link name="body">
    <visual><geometry><mesh filename="cabinet_body.glb"/></geometry></visual>
  </link>
  <joint name="door_hinge" type="revolute">
    <parent link="body"/><child link="door"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <link name="door">
    <visual><geometry><mesh filename="cabinet_door.glb"/></geometry></visual>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    cabinet = _json(
        cabinet_root / "cabinet.ao_config.json",
        {"urdf_filepath": "cabinet.urdf"},
    )

    scene = _json(
        root / "configs/scenes/apt_0.scene_instance.json",
        {
            "stage_instance": {"template_name": "stages/room"},
            "default_lighting": "lighting/frl_apartment_stage",
            "object_instances": [
                {
                    "template_name": "objects/table",
                    "translation": [1, 2, 3],
                    "rotation": [math.sqrt(0.5), 0, math.sqrt(0.5), 0],
                    "uniform_scale": 2,
                    "non_uniform_scale": [1, 0.5, 3],
                    "motion_type": "STATIC",
                },
                {
                    "template_name": "objects/chair",
                    "translation": [-1, 0, 2],
                    "rotation": [1, 0, 0, 0],
                    "motion_type": "DYNAMIC",
                },
            ],
            "articulated_object_instances": [
                {
                    "template_name": "cabinet",
                    "translation": [0, 0.5, 4],
                    "rotation": [1, 0, 0, 0],
                    "uniform_scale": 0.5,
                    "fixed_base": True,
                    "auto_clamp_joint_limits": True,
                    "initial_joint_pose": [-2],
                    "motion_type": "DYNAMIC",
                },
                {
                    "template_name": "cabinet",
                    "translation": [2, 0.5, 4],
                    "rotation": [1, 0, 0, 0],
                    "base_type": "fixed",
                    "motion_type": "DYNAMIC",
                },
            ],
        },
    )
    return {
        "root": root,
        "dataset": dataset,
        "scene": scene,
        "stage": stage,
        "chair": chair,
        "table": table,
        "cabinet": cabinet,
        "urdf": urdf,
    }


def _build(paths: dict[str, Path]):
    # Reverse the rigid config order deliberately; canonical import order must
    # be independent of caller mapping/insertion order.
    return build_replicacad_scene_plan(
        paths["dataset"],
        paths["scene"],
        stage_template_configs=[paths["stage"]],
        object_template_configs={
            "objects/table": paths["table"],
            "objects/chair": paths["chair"],
        },
        articulated_template_configs={"cabinet": paths["cabinet"]},
    )


def test_builds_count_closed_stage_rigid_and_articulated_plan(
    replica_scene: dict[str, Path],
) -> None:
    plan = _build(replica_scene)

    assert plan.coordinate_convention == COORDINATE_CONVENTION
    assert plan.default_lighting == "lighting/frl_apartment_stage"
    assert (plan.source_stage_count, plan.source_rigid_count, plan.source_articulated_count) == (
        1,
        2,
        2,
    )
    assert len(plan.stage_spawns) == 1
    assert len(plan.rigid_spawns) == 2
    assert len(plan.articulated_spawns) == 2
    assert [item.import_id for item in plan.imports] == [
        "stage:stages/room",
        "rigid:objects/chair",
        "rigid:objects/table",
        "articulated:cabinet",
    ]
    assert [spawn.spawn_id for spawn in plan.spawns] == [
        "stage:000000",
        "rigid:000000",
        "rigid:000001",
        "articulated:000000",
        "articulated:000001",
    ]
    plan.assert_closed()


def test_preserves_pbr_assets_source_pose_and_explicit_default_joint_pose(
    replica_scene: dict[str, Path],
) -> None:
    plan = _build(replica_scene)
    table = plan.rigid_spawns[0]
    articulated_import = next(
        item for item in plan.imports if item.asset_kind == "articulated"
    )

    assert table.template_name == "objects/table"
    assert table.habitat_transform.translation_m == (1.0, 2.0, 3.0)
    assert table.habitat_transform.rotation_wxyz == pytest.approx(
        (math.sqrt(0.5), 0, math.sqrt(0.5), 0)
    )
    assert table.habitat_transform.scale_xyz == (2.0, 1.0, 6.0)
    assert table.unreal_transform.translation_cm == (100.0, 300.0, 200.0)
    assert table.unreal_transform.scale_xyz == (2.0, 6.0, 1.0)
    assert articulated_import.urdf_path == replica_scene["urdf"].resolve()
    assert articulated_import.pbr_mesh_paths == (
        (replica_scene["urdf"].parent / "cabinet_body.glb").resolve(),
        (replica_scene["urdf"].parent / "cabinet_door.glb").resolve(),
    )

    clamped = plan.articulated_spawns[0].joint_defaults
    assert len(clamped) == 1
    assert clamped[0].joint_name == "door_hinge"
    assert clamped[0].position == -1.0
    assert clamped[0].source == "scene_array"
    assert clamped[0].clamped_to_limit is True

    implicit = plan.articulated_spawns[1].joint_defaults
    assert len(implicit) == 1
    assert implicit[0].position == 0.0
    assert implicit[0].source == "urdf_zero"
    assert plan.articulated_spawns[1].fixed_base is True


def test_habitat_y_up_to_unreal_centimeter_basis_is_explicit() -> None:
    assert habitat_position_to_unreal_cm([1, 2, 3]) == (100, 300, 200)
    assert habitat_scale_to_unreal([2, 3, 4]) == (2, 4, 3)
    assert habitat_quaternion_wxyz_to_unreal_xyzw(
        [math.sqrt(0.5), 0, math.sqrt(0.5), 0]
    ) == pytest.approx((0, 0, -math.sqrt(0.5), math.sqrt(0.5)))


def test_can_discover_all_selected_templates_from_dataset_config(
    replica_scene: dict[str, Path],
) -> None:
    plan = build_replicacad_scene_plan(
        replica_scene["dataset"], replica_scene["scene"]
    )

    assert len(plan.spawns) == 5
    assert len(plan.imports) == 4
    plan.assert_closed()


def test_fails_closed_for_unknown_object_template(
    replica_scene: dict[str, Path],
) -> None:
    scene = json.loads(replica_scene["scene"].read_text(encoding="utf-8"))
    scene["object_instances"][1]["template_name"] = "objects/missing_chair"
    _json(replica_scene["scene"], scene)

    with pytest.raises(ReplicaCADPlanError, match="unknown template"):
        _build(replica_scene)


def test_fails_closed_when_a_referenced_pbr_mesh_is_missing(
    replica_scene: dict[str, Path],
) -> None:
    (replica_scene["root"] / "objects/table.glb").unlink()

    with pytest.raises(ReplicaCADPlanError, match="render_asset does not exist"):
        _build(replica_scene)


def test_count_closure_detects_a_dropped_spawn(replica_scene: dict[str, Path]) -> None:
    plan = _build(replica_scene)
    broken = replace(plan, spawns=plan.spawns[:-1])

    with pytest.raises(ReplicaCADPlanError, match="not count-closed"):
        broken.assert_closed()
