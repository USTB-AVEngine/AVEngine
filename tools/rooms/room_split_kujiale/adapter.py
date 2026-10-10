"""Use recorded package transforms and existing placement navmeshes, read only."""
from __future__ import annotations
import json, math, re
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import Polygon, mapping, shape
from tools.rooms.room_selection.media import polygons
from tools.rooms.room_selection.geometry import short_side


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def transform_points(points, matrix):
    points = np.asarray(points, float)
    matrix = np.asarray(matrix, float).reshape(4, 4)
    h = np.c_[points, np.ones(len(points))] @ matrix.T
    if not np.isfinite(h).all() or np.any(np.abs(h[:, 3]) < 1e-12):
        raise ValueError('Invalid recorded source transform')
    return h[:, :3] / h[:, 3:4]


def cad_polygon(raw, matrix, floor_y):
    """rooms.json coordinates are source-world metres, without a prim transform."""
    matrix = np.asarray(matrix, float).reshape(4, 4)
    # Require a horizontal source XY plane. Never test alternative signs.
    if not np.allclose(matrix[1, :2], 0, atol=1e-10) or abs(matrix[1, 2]) < 1e-12:
        raise ValueError('Recorded transform does not map CAD XY to a horizontal floor')
    xy = np.asarray(raw['polygon'], float)
    source_z = (floor_y - matrix[1, 3]) / matrix[1, 2]
    xyz = transform_points(np.c_[xy, np.full(len(xy), source_z)], matrix)
    original = Polygon(xyz[:, [0, 2]])
    valid = original if original.is_valid else shapely.make_valid(original)
    g = shapely.union_all(list(polygons(valid)))
    if g.is_empty or not g.is_valid:
        raise ValueError('Empty or invalid CAD polygon')
    return g, dict(original_valid=original.is_valid,
                   operation='none' if original.is_valid else 'make_valid; keep polygon components',
                   source_floor_z_m=float(source_z), transform_trials=1)


def wall_axis(g):
    """Length-weighted CAD wall directions modulo the orthogonal pair."""
    ds = []
    for p in polygons(g):
        ds.extend(np.diff(np.asarray(p.exterior.coords), axis=0))
    ds = np.asarray(ds, float)
    lengths = np.linalg.norm(ds, axis=1)
    good = lengths > .1
    angles = np.arctan2(ds[good, 1], ds[good, 0])
    weights = lengths[good]
    vec = np.sum(weights * np.exp(4j * angles))
    axis = math.degrees(np.angle(vec) / 4) % 90 if len(weights) else 0.
    if abs(axis - 90) < 1e-6: axis = 0.
    return dict(primary_deg=axis, orthogonal_deg=axis + 90,
                source='length-weighted original CAD wall boundary, transformed by recorded package matrix',
                directional_concentration=float(abs(vec) / weights.sum()) if len(weights) else 0.)


def plan_inputs(old_root, dataset_root, output):
    old_root, dataset_root = Path(old_root), Path(dataset_root)
    rows = read(old_root / 'rooms_results.json')
    build_selection = read(old_root / 'evidence/build_selection.json') if (old_root / 'evidence/build_selection.json').exists() else {}
    prod_path = old_root / 'ue_round14/inputs/kujiale_v7_S_train_round18_balanced.jsonl'
    prod = [json.loads(x) for x in prod_path.read_text().splitlines() if x.strip()]
    old_keys = sorted({(r['house'], r['room_id']) for r in prod})
    checked = read(old_root / 'ue_round14/inputs/houses.json')['houses']
    checked_keys = sorted((h['house'], r) for h in checked for r in h['checked_room_ids'])
    if old_keys != checked_keys:
        raise ValueError('Production and checked-room key sets differ')
    houses = []
    for house in sorted({r['house'] for r in rows}):
        build_name = build_selection.get(house, 'build_v4')
        if '/' in build_name or not build_name.startswith('build_'): raise ValueError('Invalid build selection')
        build = old_root / 'houses' / house / build_name
        # Read the unfiltered package: the navmesh uses the unfiltered surface.
        package = build / 'acoustic_package/manifest.json'
        manifest = read(package)
        recorded = manifest['geometry']['source_to_canonical']
        if not recorded.get('reviewed'): raise ValueError('Unreviewed source transform')
        matrix = np.asarray(recorded['matrix_row_major'], float).reshape(4, 4)
        nav = build / 'navigation/navigation.navmesh'
        nav_receipt = read(build / 'navigation/build_result.json')
        if not nav.is_file() or Path(nav_receipt['navmesh']).resolve() != nav.resolve():
            raise ValueError('Placement navmesh provenance mismatch')
        semantic = read(build / 'semantic_rooms.json')['rooms']
        raw = read(dataset_root / house / 'rooms.json')
        previous = sorted([r for r in rows if r['house'] == house], key=lambda r: int(r['room_label'][1:]))
        if len(raw) != len(previous) or len(raw) != len(semantic):
            raise ValueError('CAD, geometry and room results counts differ')
        rr = []
        for i, source in enumerate(raw):
            label = 'R%d' % i
            past = next(r for r in previous if r['room_label'] == label)
            sm = next(r for r in semantic if r['room_label'] == label)
            y = float(sm['floor_height_m'])
            cad, evidence = cad_polygon(source, matrix, y)
            measured = shape(sm['floor_geometry']).buffer(0)
            outside = measured.difference(cad.buffer(.002)).area
            if outside > .01:
                raise ValueError(f'Legacy measured floor escapes recorded CAD transform: {house}/{label}: {outside}')
            # The old measured shape clips CAD to visible authored floor triangles.
            # Preserve those real holes/components rather than inventing support.
            scope = measured.intersection(cad)
            rr.append(dict(house=house, room_label=label, source_room_type=source['room_type'],
                           room_type=past['room_type'], floor_y_m=y, floor_id='F0',
                           cad_polygon_xz_m=mapping(cad), floor_polygon_xz_m=mapping(scope),
                           floor_area_m2=float(scope.area), cad_area_m2=float(cad.area),
                           cad_without_measured_floor_m2=float(cad.difference(scope).area),
                           short_side_m=short_side(scope), requires_split=scope.area > 35 + 1e-8,
                           wall_axes=wall_axis(cad), cad_transform_evidence=evidence,
                           legacy_measurement=past, legacy_floor_source=str(build / 'semantic_rooms.json'),
                           old_production_admitted=(house, label) in old_keys))
        houses.append(dict(house=house, build=str(build), navmesh=str(nav),
                           navmesh_provenance=str(build / 'navigation/build_result.json'),
                           source_stage=str(dataset_root / house / (house + '.usda')),
                           room_metadata=str(dataset_root / house / 'rooms.json'),
                           source_to_canonical=recorded, matrix=matrix.tolist(),
                           matrix_source=str(package), rooms=rr))
    result = dict(schema='kujiale_room_split_inputs_v1', houses=houses, source_rooms=len(rows),
                  source_houses=len(houses), old_production_keys=[list(k) for k in old_keys],
                  old_production_source=str(prod_path), old_production_rows=len(prod),
                  old_checked_rooms_source=str(old_root / 'ue_round14/inputs/houses.json'),
                  thresholds_source=str(old_root.parent / 'room_selection_merged_20261003/thresholds.frozen.yaml'),
                  license='InteriorAgent: noncommercial research only; no redistribution',
                  production_modified=False)
    dump(Path(output) / 'input_plan_v1.json', result)
    dump(Path(output) / 'navmesh_map_v1.json', {h['house']: h['navmesh'] for h in houses})
    return result


def coordinate_evidence(house, hs, samples=2000):
    """Area-uniform triangle samples on the actual floor-height navmesh."""
    from tools.rooms.room_selection.measurements import navmesh_triangles
    pf, polys, ys = navmesh_triangles(hs, Path(house['navmesh']))
    union = shapely.union_all([shape(r['cad_polygon_xz_m']) for r in house['rooms']])
    measured = shapely.union_all([shape(r['floor_polygon_xz_m']) for r in house['rooms']])
    floors = [r['floor_y_m'] for r in house['rooms']]
    chosen = [(p, y) for p, y in zip(polys, ys) if min(abs(y - f) for f in floors) <= .3]
    rng = np.random.default_rng(20261010)
    # Navmesh polygons are triangles, so barycentric area sampling is exact.
    ids = rng.choice(len(chosen), samples, p=np.asarray([p.area for p, y in chosen]) / sum(p.area for p, y in chosen))
    ab = rng.random((samples, 2)); flip = ab.sum(axis=1) > 1; ab[flip] = 1 - ab[flip]
    triangles = np.asarray([np.asarray(chosen[i][0].exterior.coords)[:3] for i in ids])
    xz = triangles[:, 0] + ab[:, :1] * (triangles[:, 1] - triangles[:, 0]) + ab[:, 1:] * (triangles[:, 2] - triangles[:, 0])
    exact = shapely.covers(union, shapely.points(xz))
    near = shapely.covers(union.buffer(.05), shapely.points(xz))
    # Navmesh can include furniture tops; floor height restriction is recorded.
    rec = dict(house=house['house'], matrix_source=house['matrix_source'],
               source_to_canonical=house['source_to_canonical'], transform_trials=1,
               navmesh=house['navmesh'], navmesh_provenance=house['navmesh_provenance'],
               samples=samples, seed=20261010, sampling='area-weighted barycentric navmesh triangles within .3m of measured CAD floors',
               nav_floor_area_m2=sum(p.area for p, y in chosen),
               cad_union_area_m2=float(union.area), measured_union_area_m2=float(measured.area),
               inside_cad_union_count=int(exact.sum()), inside_cad_union_fraction=float(exact.mean()),
               within_0p05m_cad_union_fraction=float(near.mean()),
               inside_measured_floor_fraction=float(shapely.covers(measured, shapely.points(xz)).mean()),
               sampled_xz_m=xz[::max(1, samples // 150)].tolist())
    # A gross mismatch stops this house; no automatic sign correction exists.
    rec['status'] = 'aligned_numeric' if near.mean() >= .9 else 'coordinate_mismatch_stop'
    return rec
