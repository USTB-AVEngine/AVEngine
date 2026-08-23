from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from avengine.m6x.geometry import RuntimeObstacleMap
from tools.m6x import build_apartment_viewpoint_bank as builder


REPOSITORY = Path(__file__).resolve().parents[2]


def _box(
    center_xyz: tuple[float, float, float],
    half_xyz: tuple[float, float, float] = (0.25, 0.5, 0.15),
) -> np.ndarray:
    center = np.asarray(center_xyz, dtype=np.float64)
    half = np.asarray(half_xyz, dtype=np.float64)
    return np.asarray(
        [
            center + np.asarray((sx, sy, sz)) * half
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )


def _m3_objects() -> list[dict[str, object]]:
    return [
        {
            "object_id": "living_table",
            "source_material_name": "MI_LivingRoom_Table",
            "vertices": _box((-3.0, 0.8, -2.0)),
        },
        {
            "object_id": "living_sofa",
            "source_material_name": "MI_Sofa",
            "vertices": _box((-2.3, 0.7, -1.8), (0.35, 0.5, 0.25)),
        },
        {
            "object_id": "kitchen_sink",
            "source_material_name": "MI_Kitchen_Sink_Base",
            "vertices": _box((2.0, 0.8, 2.0)),
        },
        {
            "object_id": "kitchen_oven",
            "source_material_name": "MI_Oven",
            "vertices": _box((2.5, 0.8, 2.1)),
        },
        {
            "object_id": "left_window_frame",
            "source_material_name": "MI_Casement",
            "vertices": _box((-5.0, 1.4, 2.0), (0.20, 0.7, 0.08)),
        },
        {
            "object_id": "left_window_glass",
            "source_material_name": "MI_Casement_Glass",
            "vertices": _box((-4.94, 1.4, 2.0), (0.18, 0.65, 0.05)),
        },
        {
            "object_id": "right_window_frame",
            "source_material_name": "MI_Casement",
            "vertices": _box((5.0, 1.4, 2.0), (0.20, 0.7, 0.08)),
        },
        {
            "object_id": "generic_fireplace_glass",
            "source_material_name": "MI_Glass",
            "vertices": _box((0.0, 0.8, 0.0)),
        },
        {
            "object_id": "ceiling_kitchen_lamp",
            "source_material_name": "MI_Ceiling_Kitchen_Lamp_Emissive",
            "vertices": _box((0.0, 2.7, 2.0)),
        },
    ]


def _anchor_library() -> dict[str, object]:
    def anchor(anchor_id: str, offset: float) -> dict[str, object]:
        return {
            "anchor_id": anchor_id,
            "kind": "entity_spawn",
            "position_m": [0.0, 0.271, 0.0],
            "los_probe_height_m": offset,
        }

    return {
        "anchors": [
            anchor("human_front_left_start", 1.63),
            anchor("human_front_left_end", 1.63),
            anchor("dog_front_right_start", 0.45),
            anchor("dog_front_right_end", 0.45),
        ]
    }


def _obstacle_map() -> RuntimeObstacleMap:
    rigid_obstacle = {
        "object_id": 17,
        "handle": "review_table",
        "obstacle_role": "elevated_object",
        "blocks_source_center": False,
        "footprint_xz_m": [
            [-5.5, -5.5],
            [-5.0, -5.5],
            [-5.0, -5.0],
            [-5.5, -5.0],
        ],
        "world_obb": {
            "center_m": [-5.25, 1.5, -5.25],
            "axes_xyz": np.eye(3).tolist(),
            "half_extents_m": [0.25, 0.25, 0.25],
        },
    }
    return RuntimeObstacleMap(
        binary_navmesh=np.ones((56, 48), dtype=np.uint8),
        bounds_m=((-6.0, 0.0, -6.0), (6.0, 3.0, 8.0)),
        floor_height_m=0.271,
        meters_per_pixel=0.25,
        rigid_obstacles=(rigid_obstacle,),
        authority="unit_test_native_loader_seam",
    )


def test_m3_targets_cluster_locally_and_do_not_average_distant_windows() -> None:
    objects = _m3_objects()
    first = builder.derive_semantic_targets_from_m3_objects(objects)
    second = builder.derive_semantic_targets_from_m3_objects(list(reversed(objects)))

    assert first == second
    by_role: dict[str, list[dict[str, object]]] = {}
    for record in first.values():
        by_role.setdefault(str(record["role"]), []).append(record)
    assert len(by_role["living_reverse"]) == 1
    assert len(by_role["kitchen_reverse"]) == 1
    assert len(by_role["window_facing"]) == 2
    window_x = sorted(
        float(record["surface_position_m"][0]) for record in by_role["window_facing"]
    )
    assert window_x == pytest.approx([-4.97, 5.0], abs=0.08)
    assert all(abs(value) > 4.5 for value in window_x)
    assert all(
        "MI_Glass" not in record["source_material_names"]
        for record in by_role["window_facing"]
    )
    assert all(
        "MI_Ceiling_Kitchen_Lamp_Emissive" not in record["source_material_names"]
        for record in first.values()
    )


def test_long_bridge_primitives_split_into_bounded_local_subsurfaces() -> None:
    window_bridge = np.concatenate(
        [
            _box((0.0, 1.4, float(z)), (0.05, 0.5, 0.05))
            for z in np.arange(-0.2, 6.61, 0.2)
        ],
        axis=0,
    )
    kitchen_bridge = np.concatenate(
        [
            _box((float(x), 0.8, -4.0), (0.05, 0.5, 0.05))
            for x in np.arange(-5.0, 5.01, 0.2)
        ],
        axis=0,
    )
    objects = [
        *_m3_objects(),
        {
            "object_id": "cross_room_window_bridge",
            "source_material_name": "MI_Casement",
            "vertices": window_bridge,
        },
        {
            "object_id": "cross_room_kitchen_bridge",
            "source_material_name": "MI_Kitchen_Extractor_Brickwall",
            "vertices": kitchen_bridge,
        },
    ]

    targets = builder.derive_semantic_targets_from_m3_objects(objects)
    bridge_target_counts = {
        "cross_room_window_bridge": 0,
        "cross_room_kitchen_bridge": 0,
    }
    for record in targets.values():
        low, high = np.asarray(record["local_component_bounds_m"], dtype=np.float64)
        span_xz = high[[0, 2]] - low[[0, 2]]
        role = str(record["role"])
        category = {
            "living_reverse": "living",
            "kitchen_reverse": "kitchen",
            "window_facing": "window",
        }[role]
        assert (
            float(np.max(span_xz)) <= builder._CATEGORY_MAX_SPAN_M[category] + 1.0e-12
        )
        for object_id in bridge_target_counts:
            if object_id in record["source_object_ids"]:
                bridge_target_counts[object_id] += 1
    assert bridge_target_counts["cross_room_window_bridge"] >= 3
    assert bridge_target_counts["cross_room_kitchen_bridge"] >= 4


def test_anchor_probe_heights_are_absolute_and_species_derived() -> None:
    result = builder.derive_anchor_probe_heights(_anchor_library())

    assert result["floor_height_m"] == pytest.approx(0.271)
    assert result["species"]["human"]["absolute_probe_heights_m"] == pytest.approx(
        [1.901]
    )
    assert result["species"]["dog"]["absolute_probe_heights_m"] == pytest.approx(
        [0.721]
    )
    assert result["corridor_source_center_heights_m"] == pytest.approx([0.721, 1.901])


def test_m3_loader_joins_manifest_material_names_to_validated_local_vertices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    vertices = _box((1.0, 2.0, 3.0))
    seen: dict[str, object] = {}

    def fake_loader(path: Path, **kwargs):
        seen["path"] = path
        seen.update(kwargs)
        return SimpleNamespace(
            manifest={
                "source_room": {
                    "room_id": "apartment0",
                    "geometry_asset_sha256": "geometry-a",
                },
                "objects": [
                    {
                        "object_id": "window0",
                        "source_material_name": "MI_Casement_Glass",
                    }
                ],
            },
            objects=({"object_id": "window0", "vertices": vertices},),
        )

    monkeypatch.setattr(builder, "load_compiled_acoustic_scene", fake_loader)
    result = builder._load_m3_surface_objects(
        manifest_path,
        room_manifest={
            "room_id": "apartment0",
            "provenance": {"exported_scene_sha256": "geometry-a"},
        },
    )

    assert seen == {
        "path": manifest_path,
        "allow_nonpassing_research_qa": True,
    }
    assert result[0]["source_material_name"] == "MI_Casement_Glass"
    assert np.array_equal(result[0]["vertices"], vertices)


@pytest.mark.parametrize(
    ("source_room", "message"),
    (
        (None, "lacks source_room binding"),
        (
            {"room_id": "wrong-room", "geometry_asset_sha256": "geometry-a"},
            "source_room.room_id differs",
        ),
        (
            {"room_id": "apartment0", "geometry_asset_sha256": "geometry-b"},
            "source_room.geometry_asset_sha256 differs",
        ),
        (
            {"room_id": "apartment0"},
            "source_room.geometry_asset_sha256 differs",
        ),
    ),
)
def test_m3_room_and_geometry_binding_fail_closed(
    source_room: object,
    message: str,
) -> None:
    manifest = {} if source_room is None else {"source_room": source_room}
    room = {
        "room_id": "apartment0",
        "provenance": {"exported_scene_sha256": "geometry-a"},
    }

    with pytest.raises(builder.ApartmentViewpointBuildError, match=message):
        builder._validate_m3_room_binding(manifest, room)


def test_build_writes_six_colocated_view0_requests_and_marked_topdown_no_clobber(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corridor_bank = tmp_path / "corridors.json"
    corridor_bank.write_text('{"segments": []}\n', encoding="utf-8")
    anchor_library = tmp_path / "anchors.json"
    anchor_library.write_text(
        json.dumps(_anchor_library(), sort_keys=True),
        encoding="utf-8",
    )
    m3_manifest = tmp_path / "m3_manifest.json"
    m3_manifest.write_text("{}\n", encoding="utf-8")
    runtime_prefix = tmp_path / "runtime-prefix"
    runtime_prefix.mkdir()
    magnum_site = tmp_path / "magnum-site"
    magnum_site.mkdir()
    monkeypatch.setenv("AVENGINE_HABITAT_MAGNUM_PYTHON_SITE", str(magnum_site))
    monkeypatch.setattr(
        builder,
        "_load_m3_surface_objects",
        lambda _path, *, room_manifest: _m3_objects(),
    )
    native_calls: list[dict[str, object]] = []

    @contextmanager
    def fake_runtime_loader(**kwargs):
        native_calls.append(kwargs)
        yield _obstacle_map()

    monkeypatch.setattr(builder, "_load_runtime_obstacle_map", fake_runtime_loader)
    output = tmp_path / "fresh-output"
    args = builder.parse_args(
        [
            "--room-manifest",
            str(
                REPOSITORY / "examples/m1/rooms/legacy_ue_apartment/room_manifest.json"
            ),
            "--base-m1-request",
            str(
                REPOSITORY
                / "examples/m6x/fixed_apartment/m1_capture_request_review_720p.json"
            ),
            "--m3-package-manifest",
            str(m3_manifest),
            "--retained-corridor-bank",
            str(corridor_bank),
            "--anchor-library",
            str(anchor_library),
            "--runtime-prefix",
            str(runtime_prefix),
            "--output",
            str(output),
        ]
    )

    assert builder.run(args) == output
    assert len(native_calls) == 1
    prefilter = json.loads((output / "prefilter.json").read_text(encoding="utf-8"))
    semantic = json.loads(
        (output / "semantic_targets.json").read_text(encoding="utf-8")
    )
    assert prefilter["artifact_status"] == "deterministic_2d_raster_prefilter_only"
    assert prefilter["studio_used"] is False
    assert prefilter["native_geometry_execution_status"] == "pass"
    assert prefilter["native_pixel_validation_status"] == "pending"
    assert "native_execution_status" not in prefilter
    assert prefilter["runtime_obstacle_map"]["rigid_obstacle_count"] == 1
    assert prefilter["runtime_obstacle_map"]["rigid_obstacles"][0]["handle"] == (
        "review_table"
    )
    assert len(prefilter["viewpoints"]) == 6
    assert set(semantic["semantic_targets"]) == {
        "living_surface_01",
        "kitchen_surface_01",
        "window_surface_01",
        "window_surface_02",
    }

    request_ids = set()
    for viewpoint in prefilter["viewpoints"]:
        request_path = output / prefilter["request_paths"][viewpoint["viewpoint_id"]]
        request = json.loads(request_path.read_text(encoding="utf-8"))
        rig = request["primary_camera_rig"]
        assert rig["view_id"] == "view0"
        assert rig["world_from_rig"]["translation_m"] == pytest.approx(
            viewpoint["camera"]["habitat_translation_m"]
        )
        assert request["listener"] == {
            "attached_to": "camera_rig_0",
            "listener_id": "listener0",
            "rig_from_listener": {
                "rotation_xyzw": [0, 0, 0, 1],
                "translation_m": [0, 0, 0],
            },
        }
        request_ids.add(request["request_id"])
    assert len(request_ids) == 6

    topdown_path = output / "topdown_2d_prefilter_native_pending.png"
    with Image.open(topdown_path) as image:
        assert image.size == (1500, 1100)
        assert image.info["artifact_status"] == (
            "deterministic_2d_raster_prefilter_only"
        )
        assert image.info["native_pixel_validation_status"] == "pending"
        assert image.info["claim_boundary"] == builder.PREFILTER_LABEL

    with pytest.raises(
        builder.ApartmentViewpointBuildError,
        match="refusing to replace output or staging path",
    ):
        builder.run(args)
    assert len(native_calls) == 1

    race_output = tmp_path / "race-output"
    race_args = builder.parse_args(
        [
            "--room-manifest",
            str(args.room_manifest),
            "--base-m1-request",
            str(args.base_m1_request),
            "--m3-package-manifest",
            str(args.m3_package_manifest),
            "--retained-corridor-bank",
            str(args.retained_corridor_bank),
            "--anchor-library",
            str(args.anchor_library),
            "--runtime-prefix",
            str(args.runtime_prefix),
            "--output",
            str(race_output),
        ]
    )
    original_publish = builder.atomic_publish_directory

    def racing_publish(policy, staging, destination):
        destination.mkdir()
        return original_publish(policy, staging, destination)

    monkeypatch.setattr(builder, "atomic_publish_directory", racing_publish)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        builder.run(race_args)
    assert race_output.is_dir()
    assert list(race_output.iterdir()) == []
    assert (tmp_path / ".race-output.staging" / "prefilter.json").is_file()
    assert len(native_calls) == 2


def test_runtime_loader_fails_when_declared_navmesh_cannot_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    navmesh = tmp_path / "apartment.navmesh"
    navmesh.write_bytes(b"declared-navmesh")

    class FakeSimulator:
        def __init__(self) -> None:
            self.pathfinder = SimpleNamespace(load_nav_mesh=lambda _path: False)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    runtime = SimpleNamespace(
        physics_config_path=tmp_path / "default.physics_config.json",
        habitat_sim=SimpleNamespace(Simulator=lambda _configuration: FakeSimulator()),
        magnum=object(),
    )
    monkeypatch.setattr(
        builder,
        "prepare_installed_habitat_runtime",
        lambda **_kwargs: runtime,
    )
    monkeypatch.setattr(
        builder,
        "_make_configuration",
        lambda *_args, **_kwargs: (
            object(),
            {},
            "listener0",
            {
                "navmesh_policy": "load_declared",
                "navmesh": navmesh,
            },
        ),
    )

    with pytest.raises(
        builder.ApartmentViewpointBuildError,
        match="could not load the explicitly declared Apartment navmesh",
    ):
        with builder._load_runtime_obstacle_map(
            inputs=SimpleNamespace(),
            runtime_prefix=tmp_path / "runtime",
            magnum_python_site=tmp_path / "magnum",
            floor_height_m=0.271,
            meters_per_pixel=0.04,
            scratch_directory=tmp_path / "scratch",
        ):
            pytest.fail("unloadable declared navmesh unexpectedly yielded a map")


def test_magnum_site_is_explicit_and_has_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AVENGINE_HABITAT_MAGNUM_PYTHON_SITE", raising=False)

    with pytest.raises(
        FileNotFoundError,
        match="AVENGINE_HABITAT_MAGNUM_PYTHON_SITE must be explicitly set",
    ):
        builder._required_magnum_python_site()
