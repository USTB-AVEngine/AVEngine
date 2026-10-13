"""Collision component vertex exposure must retain the real chair geometry."""
from pathlib import Path
import numpy as np
import trimesh
from avengine.assets.roster_motion.chair_collision import prepare_chair_collision
from avengine.assets.roster_motion.io import read_glb, write_glb
from avengine.capture.v7_visible_mesh import indexed_static_mesh


def original_parts(path: Path) -> bytes:
    left = trimesh.creation.box(extents=[.4, .5, .6])
    right = trimesh.creation.box(extents=[.3, .7, .2])
    left.apply_translation([-2, .25, 0])
    right.apply_translation([2, .35, 0])
    path.write_bytes(trimesh.util.concatenate([left, right]).export(file_type='glb'))
    doc, blob = read_glb(path)
    for node in doc['nodes']:
        if 'mesh' in node:
            node['scale'] = [1, .91, 1]
    write_glb(path, doc, blob)
    return path.read_bytes()


def positions(doc, blob, accessor):
    a = doc['accessors'][accessor]
    view = doc['bufferViews'][a['bufferView']]
    return np.ndarray((a['count'], 3), dtype='<f4', buffer=blob,
        offset=view.get('byteOffset', 0) + a.get('byteOffset', 0),
        strides=(view.get('byteStride', 12), 4)).copy()


def triangles(path):
    mesh = indexed_static_mesh(path)
    return sorted(tuple(float(x) for x in row.ravel())
        for row in np.asarray(mesh.vertices)[np.asarray(mesh.faces)])


def test_unreferenced_other_part_vertices_cannot_fill_the_chair_gap(tmp_path):
    source = tmp_path / 'original.glb'
    original_parts(source)
    target = tmp_path / 'collision.glb'
    prepare_chair_collision(source, target)
    doc, blob = read_glb(target)
    parts = doc['meshes'][0]['primitives']
    assert len(parts) == 2
    bounds = []
    for part in parts:
        vertices = positions(doc, blob, part['attributes']['POSITION'])
        bounds.append((vertices[:, 0].min(), vertices[:, 0].max()))
        assert vertices[:, 0].max() < -1 or vertices[:, 0].min() > 1
    assert min(v[0] for v in bounds) < -1
    assert max(v[1] for v in bounds) > 1


def test_all_original_triangles_transforms_materials_and_input_bytes_survive(tmp_path):
    source = tmp_path / 'original.glb'
    original = original_parts(source)
    target = tmp_path / 'collision.glb'
    prepare_chair_collision(source, target)
    before, _ = read_glb(source)
    after, _ = read_glb(target)
    assert source.read_bytes() == original
    assert before['nodes'] == after['nodes']
    assert before.get('materials') == after.get('materials')
    assert triangles(source) == triangles(target)
    original_time = target.stat().st_mtime_ns
    prepare_chair_collision(source, target)
    assert target.stat().st_mtime_ns == original_time
