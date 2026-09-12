"""Expected-upload-report reuse and the vectorized world-geometry canonicalizer.

Deriving one compiled scene's expected RLR upload report canonicalizes every
world-space vertex and triangle.  Two things are asserted here:

* the batched canonicalizer returns exactly the bytes the original per-vertex
  form returned, so ``expected_world_geometry_sha1`` still matches the native
  upload; and
* :class:`~avengine.acoustics.runtime.CompiledSceneUploadExpectation` may share
  that derivation across several native contexts over one unchanged scene,
  while every real upload is still compared field by field, and any change to
  the scene's mutable containers makes the expectation refuse to serve.

These are hermetic Python contracts.  They do not stand in for the native RLR
upload comparison recorded alongside this task.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from avengine.acoustics.runtime import (
    CompiledAcousticScene,
    CompiledSceneUploadExpectation,
    RuntimeContractError,
    _canonical_world_geometry,
    _expected_upload_report,
    _verify_upload_report,
)


def _reference_canonical_world_geometry(scene: CompiledAcousticScene) -> bytes:
    """The pre-optimization per-vertex form, kept as the exactness reference."""

    def coordinate(value: float) -> str:
        number = float(value)
        if not math.isfinite(number):
            raise RuntimeContractError("scene geometry contains a non-finite value")
        if abs(number) < 0.5e-6:
            number = 0.0
        return f"{number:.6f}"

    vertex_tokens: list[str] = []
    triangle_tokens: list[str] = []
    for item in scene.objects:
        position = np.asarray(item["position"], dtype=np.float64)
        quaternion = np.asarray(item["orientation_wxyz"], dtype=np.float64)
        w = float(quaternion[0])
        q = quaternion[1:]
        object_tokens: list[str] = []
        for raw_vertex in np.asarray(item["vertices"], dtype=np.float64):
            twice_cross = 2.0 * np.cross(q, raw_vertex)
            transformed = raw_vertex + w * twice_cross + np.cross(q, twice_cross)
            transformed = transformed + position
            token = " ".join(coordinate(value) for value in transformed)
            object_tokens.append(token)
            vertex_tokens.append(token)
        for face in np.asarray(item["triangles"], dtype=np.int64):
            values = [object_tokens[int(index)] for index in face]
            rotations = [
                "|".join(values),
                "|".join(values[1:] + values[:1]),
                "|".join(values[2:] + values[:2]),
            ]
            triangle_tokens.append(min(rotations))
    lines = ["AVENGINE_RLR_WORLD_GEOMETRY_V1"]
    lines.extend(f"v {value}" for value in sorted(vertex_tokens))
    lines.extend(f"f {value}" for value in sorted(triangle_tokens))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _material(name: str, label: str, offset: float) -> dict[str, object]:
    def band(base: float) -> list[float]:
        return [125.0, base, 250.0, base + 0.05, 500.0, base + 0.1, 1000.0, base + 0.15]

    return {
        "name": name,
        "labels": [label],
        "absorption": band(0.10 + offset),
        "scattering": band(0.20 + offset),
        "transmission": band(0.05 + offset),
    }


def _object(
    object_id: str,
    *,
    vertices: np.ndarray,
    triangles: np.ndarray,
    material_ids: np.ndarray,
    position: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
) -> dict[str, object]:
    return {
        "object_id": object_id,
        "position": list(position),
        "orientation_wxyz": list(orientation_wxyz),
        "vertices": np.ascontiguousarray(vertices, dtype="<f4"),
        "triangles": np.ascontiguousarray(triangles, dtype="<u4"),
        "triangle_material_ids": np.ascontiguousarray(material_ids, dtype="<u4"),
    }


def _scene(*, seed: int = 7, shift: float = 0.0) -> CompiledAcousticScene:
    """One small two-object scene with awkward but legal coordinates."""

    generator = np.random.default_rng(seed)
    wall_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [-0.0, 2.5e-7, -3.0e-7],  # below the 0.5e-6 flush-to-zero threshold
            [1.0, 0.0, 0.0],
            [1.0, 2.4, 0.0],
            [0.0, 2.4, 0.0],
            [0.5, 1.2, 0.25],
        ],
        dtype=np.float64,
    ) + shift
    wall_triangles = np.array(
        [[0, 2, 3], [0, 3, 4], [1, 5, 2], [3, 4, 0]], dtype=np.int64
    )
    floor_vertices = generator.uniform(-3.0, 3.0, size=(24, 3))
    floor_triangles = np.stack(
        [
            np.arange(0, 21, dtype=np.int64),
            np.arange(1, 22, dtype=np.int64),
            np.arange(2, 23, dtype=np.int64),
        ],
        axis=1,
    )
    axis = np.array([0.3, -0.7, 0.6])
    axis = axis / np.linalg.norm(axis)
    angle = 0.9
    quaternion = (
        float(np.cos(angle / 2.0)),
        *(float(value) for value in axis * np.sin(angle / 2.0)),
    )
    objects = (
        _object(
            "wall_object",
            vertices=wall_vertices,
            triangles=wall_triangles,
            material_ids=np.zeros(len(wall_triangles), dtype=np.int64),
            position=(0.125, -1.5, 2.0),
            orientation_wxyz=quaternion,
        ),
        _object(
            "floor_object",
            vertices=floor_vertices,
            triangles=floor_triangles,
            material_ids=np.array(
                [index % 2 for index in range(len(floor_triangles))], dtype=np.int64
            ),
            position=(-0.75, 0.0, 0.5),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    database = b'{"materials": "fixture"}'
    return CompiledAcousticScene(
        manifest_path=__file__,
        manifest_sha256="a" * 64,
        manifest={},
        package_id="p14_upload_reuse_fixture",
        package_content_sha256="b" * 64,
        material_database_path=__file__,
        material_database_bytes=database,
        material_database_sha256=hashlib.sha256(database).hexdigest(),
        material_categories_document={},
        rlr_material_database={
            "materials": [
                _material("concrete_wall", "wall", 0.0),
                _material("wooden_floor", "floor", 0.03),
            ]
        },
        material_categories=("wall", "floor"),
        objects=objects,
        geometry_records={},
        triangle_count_by_material={
            "wall": 4 + len(floor_triangles) - len(floor_triangles) // 2,
            "floor": len(floor_triangles) // 2,
        },
        qa_reports={},
    )


def test_batched_world_geometry_matches_the_per_vertex_reference_bytes() -> None:
    scene = _scene()

    produced = _canonical_world_geometry(scene)

    assert produced == _reference_canonical_world_geometry(scene)
    text = produced.decode("utf-8")
    assert text.startswith("AVENGINE_RLR_WORLD_GEOMETRY_V1\n")
    # The sub-threshold coordinates are flushed to a positive zero token.
    assert "\nv 0.125000 -1.500000 2.000000\n" in "\n" + text
    assert "-0.000000" not in text


def test_empty_and_non_finite_objects_keep_their_original_behaviour() -> None:
    scene = _scene()
    empty = _object(
        "empty_object",
        vertices=np.zeros((0, 3)),
        triangles=np.zeros((0, 3)),
        material_ids=np.zeros(0),
        position=(0.0, 0.0, 0.0),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    with_empty = CompiledAcousticScene(
        **{
            **{
                field: getattr(scene, field)
                for field in scene.__dataclass_fields__
            },
            "objects": scene.objects + (empty,),
        }
    )

    assert _canonical_world_geometry(with_empty) == _canonical_world_geometry(scene)

    broken = dict(scene.objects[0])
    vertices = np.array(broken["vertices"], dtype=np.float64)
    vertices[2, 1] = np.inf
    broken["vertices"] = vertices
    non_finite = CompiledAcousticScene(
        **{
            **{
                field: getattr(scene, field)
                for field in scene.__dataclass_fields__
            },
            "objects": (broken,) + scene.objects[1:],
        }
    )
    with np.errstate(invalid="ignore"):
        with pytest.raises(RuntimeContractError, match="non-finite"):
            _canonical_world_geometry(non_finite)
        with pytest.raises(RuntimeContractError, match="non-finite"):
            _reference_canonical_world_geometry(non_finite)


def test_precomputed_expectation_reproduces_the_direct_expected_report() -> None:
    scene = _scene()
    direct = _expected_upload_report(scene)

    expectation = CompiledSceneUploadExpectation(scene)

    assert expectation.scene is scene
    assert expectation.expected_report() == direct
    assert expectation.expected_report(scene) == direct
    # Each caller receives a private copy, so one caller cannot poison another.
    borrowed = expectation.expected_report()
    borrowed["vertex_count"] = -1
    borrowed["material_upload_receipts"].clear()
    assert expectation.expected_report() == direct


def test_reused_expectation_still_compares_every_actual_upload_field() -> None:
    scene = _scene()
    expectation = CompiledSceneUploadExpectation(scene)
    actual = _expected_upload_report(scene)

    # Repeated verification of the same real upload keeps passing.
    for _ in range(3):
        _verify_upload_report(scene, actual, expectation=expectation)
        _verify_upload_report(scene, actual)

    for field, replacement in (
        ("vertex_count", 0),
        ("triangle_count", 1),
        ("expected_world_geometry_sha1", "c" * 40),
        ("expected_material_coefficient_sha1", "d" * 40),
        ("material_upload_receipts", []),
        ("resolved_material_index_by_category", {"wall": 1, "floor": 0}),
    ):
        wrong = {**actual, field: replacement}
        with pytest.raises(RuntimeContractError, match=field):
            _verify_upload_report(scene, wrong, expectation=expectation)
        with pytest.raises(RuntimeContractError, match=field):
            _verify_upload_report(scene, wrong)

    missing = {name: value for name, value in actual.items() if name != "object_ids"}
    with pytest.raises(RuntimeContractError, match="object_ids"):
        _verify_upload_report(scene, missing, expectation=expectation)


def test_expectation_refuses_a_scene_whose_mutable_containers_changed() -> None:
    scene = _scene()
    expectation = CompiledSceneUploadExpectation(scene)
    actual = _expected_upload_report(scene)
    item = scene.objects[0]

    original_vertices = item["vertices"]
    moved = np.array(original_vertices, copy=True)
    moved[0, 0] = np.float32(float(moved[0, 0]) + 0.25)
    item["vertices"] = moved
    with pytest.raises(RuntimeContractError, match="changed after"):
        _verify_upload_report(scene, actual, expectation=expectation)
    # The changed scene really does produce a different expected report, so the
    # refusal above prevented a wrong reuse rather than a spurious one.
    assert _expected_upload_report(scene) != actual
    item["vertices"] = original_vertices
    _verify_upload_report(scene, actual, expectation=expectation)

    original_triangles = item["triangles"]
    item["triangles"] = np.ascontiguousarray(original_triangles[:-1])
    with pytest.raises(RuntimeContractError, match="changed after"):
        _verify_upload_report(scene, actual, expectation=expectation)
    item["triangles"] = original_triangles
    _verify_upload_report(scene, actual, expectation=expectation)

    original_orientation = item["orientation_wxyz"]
    item["orientation_wxyz"] = [0.0, 1.0, 0.0, 0.0]
    with pytest.raises(RuntimeContractError, match="changed after"):
        _verify_upload_report(scene, actual, expectation=expectation)
    item["orientation_wxyz"] = original_orientation
    _verify_upload_report(scene, actual, expectation=expectation)

    material = scene.rlr_material_database["materials"][0]
    original_absorption = list(material["absorption"])
    material["absorption"] = [
        value + (0.01 if index % 2 else 0.0)
        for index, value in enumerate(original_absorption)
    ]
    with pytest.raises(RuntimeContractError, match="changed after"):
        _verify_upload_report(scene, actual, expectation=expectation)
    assert _expected_upload_report(scene) != actual
    material["absorption"] = original_absorption
    _verify_upload_report(scene, actual, expectation=expectation)

    scene.triangle_count_by_material["wall"] += 1
    with pytest.raises(RuntimeContractError, match="changed after"):
        _verify_upload_report(scene, actual, expectation=expectation)
    scene.triangle_count_by_material["wall"] -= 1
    _verify_upload_report(scene, actual, expectation=expectation)


def test_expectation_is_bound_to_one_scene_object_not_to_its_address() -> None:
    scene = _scene()
    twin = _scene()
    other = _scene(shift=0.5)
    expectation = CompiledSceneUploadExpectation(scene)

    assert _expected_upload_report(twin) == _expected_upload_report(scene)
    assert _expected_upload_report(other) != _expected_upload_report(scene)

    # Even a content-equal scene is a different caller scope and is refused.
    for foreign in (twin, other):
        with pytest.raises(RuntimeContractError, match="different compiled scene"):
            expectation.verify(foreign, _expected_upload_report(foreign))
        with pytest.raises(RuntimeContractError, match="different compiled scene"):
            expectation.expected_report(foreign)

    # A fresh expectation for the changed scene is required, and it works.
    _verify_upload_report(
        other, _expected_upload_report(other),
        expectation=CompiledSceneUploadExpectation(other),
    )


def test_expectation_requires_a_validated_compiled_scene() -> None:
    with pytest.raises(RuntimeContractError, match="CompiledAcousticScene"):
        CompiledSceneUploadExpectation(object())
