"""HM3D semantics: where the identity lives, and every way it silently breaks."""

from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
import pytest

from avengine.acoustics.semantic import (
    HM3D_UNANNOTATED_CATEGORY,
    SemanticSceneError,
    load_hm3d_semantic_scene,
    parse_hm3d_annotation_bytes,
)


def _srgb_byte_to_u16(byte_value: int) -> int:
    """Encode an sRGB byte the way HM3D's mesh carries it: linear, 16 bit."""

    channel = byte_value / 255.0
    linear = (
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
    )
    return int(round(linear * 65535.0))


def _coloured_glb(
    *,
    triangles: list[tuple[list[list[float]], list[tuple[int, int, int]]]],
    node_to_mesh: list[int] | None = None,
) -> bytes:
    """Build a minimal semantic GLB: one mesh per triangle, COLOR_0 per vertex.

    Each entry of ``triangles`` is (three positions, three sRGB byte colours),
    so a caller can paint the vertices of one triangle differently and see how
    the loader resolves it.
    """

    meshes: list[dict] = []
    accessors: list[dict] = []
    views: list[dict] = []
    binary = b""
    for positions, colours in triangles:
        position_array = np.asarray(positions, dtype="<f4")
        colour_array = np.asarray(
            [
                [_srgb_byte_to_u16(channel) for channel in colour] + [65535]
                for colour in colours
            ],
            dtype="<u2",
        )
        index_array = np.asarray([0, 1, 2], dtype="<u2")
        for array, accessor in (
            (position_array, {"componentType": 5126, "type": "VEC3", "count": 3}),
            (
                colour_array,
                {
                    "componentType": 5123,
                    "type": "VEC4",
                    "count": 3,
                    "normalized": True,
                },
            ),
            (index_array, {"componentType": 5123, "type": "SCALAR", "count": 3}),
        ):
            payload = array.tobytes()
            binary += b"\x00" * ((-len(binary)) % 4)
            views.append(
                {"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload)}
            )
            accessors.append({**accessor, "bufferView": len(views) - 1})
            binary += payload
        base = len(accessors) - 3
        meshes.append(
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": base, "COLOR_0": base + 1},
                        "indices": base + 2,
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        )
    declared = len(binary)
    binary += b"\x00" * ((-len(binary)) % 4)
    if node_to_mesh is None:
        node_to_mesh = list(range(len(meshes)))
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(node_to_mesh)))}],
        "nodes": [{"mesh": index} for index in node_to_mesh],
        "meshes": meshes,
        "materials": [{"name": "SemanticPaint"}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": declared}],
    }
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    return (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def _annotation(lines: list[str]) -> bytes:
    return ("HM3D Semantic Annotations\n" + "\n".join(lines) + "\n").encode("utf-8")


def _write(tmp_path: Path, glb: bytes, annotations: bytes) -> tuple[Path, Path]:
    glb_path = tmp_path / "scene.semantic.glb"
    text_path = tmp_path / "scene.semantic.txt"
    glb_path.write_bytes(glb)
    text_path.write_bytes(annotations)
    return glb_path, text_path


UNIT = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_annotation_maps_colour_to_canonical_category() -> None:
    parsed = parse_hm3d_annotation_bytes(
        _annotation(['1,97C517,"ceiling",1', '2,07576C,"Sofa Chair",1'])
    )
    assert parsed.colour_to_category[(0x97, 0xC5, 0x17)] == "ceiling"
    assert parsed.colour_to_category[(0x07, 0x57, 0x6C)] == "sofa_chair"
    assert parsed.colour_to_raw_category[(0x07, 0x57, 0x6C)] == "Sofa Chair"
    assert parsed.colour_to_instance_id[(0x07, 0x57, 0x6C)] == 2
    assert parsed.defects == ()


def test_one_colour_naming_two_categories_is_rejected() -> None:
    with pytest.raises(SemanticSceneError, match="names two categories"):
        parse_hm3d_annotation_bytes(
            _annotation(['1,97C517,"ceiling",1', '2,97C517,"floor",1'])
        )


def test_a_single_malformed_line_is_skipped_and_recorded() -> None:
    """HM3D ships exactly this defect: 474,c,"radiator",11 in 00546-nS8T59Aw3sf."""

    lines = [f'{index},{index:02X}{index:02X}{index:02X},"wall",1' for index in range(1, 120)]
    lines.append('474,c,"radiator",11')
    parsed = parse_hm3d_annotation_bytes(_annotation(lines))
    assert len(parsed.colour_to_category) == 119
    assert parsed.defects == ("line 121: colour 'c' is not six hex digits",)


def test_a_mostly_malformed_file_is_rejected() -> None:
    lines = ['1,97C517,"ceiling",1'] + [f'{index},zz,"wall",1' for index in range(2, 10)]
    with pytest.raises(SemanticSceneError, match="above the"):
        parse_hm3d_annotation_bytes(_annotation(lines))


def test_the_srgb_transfer_curve_is_what_matches_the_annotation(tmp_path: Path) -> None:
    """Reading COLOR_0 as if it were already sRGB matches nothing at all.

    This is the failure that reported every face of a real scene unannotated:
    zero of 395018 faces matched, with no error anywhere. The guard is that the
    encoded value and the annotation key are deliberately far apart, so a loader
    skipping the transfer curve cannot accidentally still match.
    """

    colour = (0x97, 0xC5, 0x17)
    glb_path, text_path = _write(
        tmp_path,
        _coloured_glb(triangles=[(UNIT, [colour] * 3)]),
        _annotation(['1,97C517,"ceiling",1']),
    )
    scene = load_hm3d_semantic_scene(glb_path, text_path)
    assert scene.semantic_categories == ("ceiling",)

    raw_u16 = [_srgb_byte_to_u16(channel) for channel in colour]
    as_if_srgb = tuple(int(round(value / 65535.0 * 255.0)) for value in raw_u16)
    assert as_if_srgb != colour


def test_unpainted_black_is_labelled_rather_than_dropped(tmp_path: Path) -> None:
    """Unannotated geometry still reflects sound, so it keeps its triangles."""

    offset = [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]]
    glb_path, text_path = _write(
        tmp_path,
        _coloured_glb(
            triangles=[
                (UNIT, [(0x97, 0xC5, 0x17)] * 3),
                (offset, [(0, 0, 0)] * 3),
            ]
        ),
        _annotation(['1,97C517,"ceiling",1']),
    )
    scene = load_hm3d_semantic_scene(glb_path, text_path)
    assert len(scene.triangles) == 2
    assert set(scene.semantic_categories) == {"ceiling", HM3D_UNANNOTATED_CATEGORY}
    assert scene.category_triangle_counts[HM3D_UNANNOTATED_CATEGORY] == 1


def test_two_agreeing_vertices_decide_a_mixed_triangle(tmp_path: Path) -> None:
    ceiling = (0x97, 0xC5, 0x17)
    floor = (0x07, 0x57, 0x6C)
    glb_path, text_path = _write(
        tmp_path,
        _coloured_glb(triangles=[(UNIT, [ceiling, floor, floor])]),
        _annotation(['1,97C517,"ceiling",1', '2,07576C,"floor",1']),
    )
    scene = load_hm3d_semantic_scene(glb_path, text_path)
    assert scene.semantic_categories == ("floor",)


def test_colour_follows_the_object_record_not_the_walk_order(tmp_path: Path) -> None:
    """The two orders are made to disagree, because when they do it is silent.

    Node 0 points at mesh 1 and node 1 at mesh 0, so a reader that pairs the
    Nth walked primitive with the Nth mesh assigns both triangles the other
    one's material and produces a scene that validates and is wrong.
    """

    ceiling = (0x97, 0xC5, 0x17)
    floor = (0x07, 0x57, 0x6C)
    low = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    high = [[0.0, 4.0, 0.0], [1.0, 4.0, 0.0], [0.0, 4.0, 1.0]]
    glb_path, text_path = _write(
        tmp_path,
        _coloured_glb(
            triangles=[(low, [floor] * 3), (high, [ceiling] * 3)],
            node_to_mesh=[1, 0],
        ),
        _annotation(['1,97C517,"ceiling",1', '2,07576C,"floor",1']),
    )
    scene = load_hm3d_semantic_scene(glb_path, text_path)
    heights = {}
    for record in scene.objects:
        block = scene.vertices[
            record["vertex_offset"] : record["vertex_offset"] + record["vertex_count"]
        ]
        heights[record["source_material_name"]] = float(block[:, 1].mean())
    assert heights["ceiling"] == pytest.approx(4.0)
    assert heights["floor"] == pytest.approx(0.0)


def test_grouping_preserves_every_triangle_and_its_area(tmp_path: Path) -> None:
    ceiling = (0x97, 0xC5, 0x17)
    floor = (0x07, 0x57, 0x6C)
    shapes = [
        ([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], [floor] * 3),
        ([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 5.0, 1.0]], [ceiling] * 3),
        ([[0.0, 0.0, 2.0], [4.0, 0.0, 2.0], [0.0, 1.0, 2.0]], [floor] * 3),
    ]
    glb_path, text_path = _write(
        tmp_path,
        _coloured_glb(triangles=shapes),
        _annotation(['1,97C517,"ceiling",1', '2,07576C,"floor",1']),
    )
    scene = load_hm3d_semantic_scene(glb_path, text_path)
    assert len(scene.triangles) == 3
    assert scene.category_triangle_counts == {"ceiling": 1, "floor": 2}

    def area(vertices: np.ndarray, faces: np.ndarray) -> float:
        a = vertices[faces[:, 0]].astype(float)
        b = vertices[faces[:, 1]].astype(float)
        c = vertices[faces[:, 2]].astype(float)
        return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2.0)

    expected = sum(
        area(np.asarray(positions, dtype="<f4"), np.asarray([[0, 1, 2]]))
        for positions, _ in shapes
    )
    assert area(scene.vertices, scene.triangles.astype(np.int64)) == pytest.approx(
        expected
    )
