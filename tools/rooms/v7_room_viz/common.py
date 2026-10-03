"""Read-only evidence loading and compact display preparation."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict
import numpy as np
from shapely.geometry import shape, box, mapping
from shapely.ops import unary_union


def now():
    return datetime.now(ZoneInfo('Asia/Singapore')).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def dominant(row):
    floors = row.get('floors', [])
    selected = (row.get('floor_selection') or {}).get('selected_floor_id')
    if floors:
        return next((f['metrics'] for f in floors if f['floor_id'] == selected),
                    max(floors, key=lambda f: f['metrics']['floor_area_m2'])['metrics'])
    metrics = row.get('metrics') or {}
    return metrics if metrics.get('floor_polygon') else None


def display_levels(rows, house):
    """Use the demo's native MP3D levels / HM3D height grouping, plus witness cuts.

    Height slices and registry proxies are explicitly not certified story counts.
    Display simplification never feeds back into acoustic measurements.
    """
    items = []
    for row in rows:
        metrics = dominant(row)
        native = row.get('native_region') or {}
        legacy = row.get('legacy_room') or {}
        if metrics:
            geom = shape(metrics['floor_polygon'])
            height = metrics['floor_y_m']
        else:
            bounds = native.get('bbox_xz_m') or legacy.get('bbox_xz_m')
            if not bounds:
                continue
            geom = box(*np.asarray(bounds).ravel())
            height = native.get('floor_y_m', legacy.get('floor_y_m'))
            if height is None:
                continue
        if geom.is_empty:
            continue
        items.append(dict(room_label=row['room_label'], floor_y_m=float(height),
                          geometry=geom, level_index=native.get('level_index'),
                          measured_floor=metrics is not None))
    if not items:
        for room in house['acoustic_rooms']:
            items.append(dict(room_label=room['room_label'],
                              floor_y_m=room['placement']['camera_m'][1] - 1.5,
                              geometry=shape(room['floor_polygon']),
                              level_index=None, measured_floor=True))
    if house['family'] == 'mp3d' and any(x['level_index'] is not None for x in items):
        groups = [[x for x in items if x['level_index'] == k]
                  for k in sorted({x['level_index'] for x in items if x['level_index'] is not None})]
    else:
        groups = []
        for item in sorted(items, key=lambda x: x['floor_y_m']):
            if not groups or item['floor_y_m'] - groups[-1][0]['floor_y_m'] > 1.5:
                groups.append([])
            groups[-1].append(item)
    levels = []
    for i, group in enumerate(groups):
        weighted = sorted((x['floor_y_m'], x['geometry'].area) for x in group)
        cumulative = 0
        total = sum(w for _, w in weighted)
        fy = weighted[0][0]
        for y, w in weighted:
            cumulative += w
            if cumulative >= total / 2:
                fy = y
                break
        levels.append(dict(level_id=i, kind='registry_level', floor_y_m=fy,
                           room_labels=[x['room_label'] for x in group],
                           geometry=unary_union([x['geometry'] for x in group]).buffer(0),
                           outlines=group, acoustic_room_labels=[]))
    for room in house['acoustic_rooms']:
        witness_y = room['placement']['camera_m'][1] - 1.5
        nearest = min(levels, key=lambda x: abs(x['floor_y_m'] - witness_y))
        if abs(nearest['floor_y_m'] - witness_y) > .45:
            supplement = next((x for x in levels if x['kind'] == 'witness_height' and
                               abs(x['floor_y_m'] - witness_y) <= .3), None)
            if supplement is None:
                supplement = dict(level_id=len(levels), kind='witness_height',
                                  floor_y_m=float(witness_y), room_labels=[],
                                  geometry=nearest['geometry'], outlines=nearest['outlines'],
                                  acoustic_room_labels=[])
                levels.append(supplement)
            nearest = supplement
        nearest['acoustic_room_labels'].append(room['room_label'])
        nearest['geometry'] = unary_union([nearest['geometry'], shape(room['floor_polygon'])]).buffer(0)
    return levels


def prepare_jobs(input_root, output):
    root, output = Path(input_root), Path(output)
    manifest = read_json(root / 'acceptance_manifest_v2.json')
    wanted = {h['house']: h for h in manifest['houses']}
    registries = {h['registry_source'] for h in wanted.values()}
    rows = defaultdict(list)
    for registry in registries:
        with Path(registry).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['house'] in wanted:
                    rows[row['house']].append(row)
    jobs = []
    for house_id, h in wanted.items():
        result_path = root / 'measurement_final' / 'houses' / house_id / 'result.json'
        result = read_json(result_path)
        visual_path = result_path.with_name('visual_checks.json')
        visual = read_json(visual_path) if visual_path.exists() else {}
        boundary_path = root / 'geometry_review' / 'houses' / house_id / 'boundary_review.json'
        boundary = read_json(boundary_path) if boundary_path.exists() else {}
        levels = display_levels(rows[house_id], h)
        serial = []
        for level in levels:
            v = dict(level)
            v['geometry'] = mapping(v['geometry'].simplify(.02, preserve_topology=True))
            v['outlines'] = [{**x, 'geometry': mapping(x['geometry'].simplify(.02, preserve_topology=True))}
                             for x in v['outlines']]
            serial.append(v)
        source = next((r.get('collision_source') for r in rows[house_id] if r.get('collision_source')), None)
        source = source or str(Path(h['scene_directory']) / (h['scan_id'] + '.glb'))
        job = dict(house=h, result=result, visual=visual, boundary=boundary,
                   levels=serial, source=source, input_paths=dict(
                       result=str(result_path), visual=str(visual_path), boundary=str(boundary_path),
                       manifest=str(root / 'acceptance_manifest_v2.json'), registry=h['registry_source']))
        path = output / 'jobs' / (house_id + '.json')
        write_json(path, job)
        jobs.append(path)
    write_json(output / 'evidence' / 'prepared.json', dict(at=now(), houses=len(jobs),
               display_polygon_simplification_m=.02, source_root=str(root), jobs=[str(p) for p in jobs]))
    return jobs


def project(point, view, projection, width, height):
    clip = np.asarray(projection) @ np.asarray(view) @ np.r_[point, 1.0]
    if clip[3] <= 0:
        return None
    ndc = clip[:3] / clip[3]
    return ((ndc[0] + 1) * width / 2, (1 - ndc[1]) * height / 2, float(ndc[2]))
