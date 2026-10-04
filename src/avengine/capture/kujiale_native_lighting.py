"""Native light pose assertions for the USD-to-UE Y reflection adapter."""
from __future__ import annotations
import math


def reflect_native_light_pose(position_cm, rotation_pitch_yaw_roll):
    """Mirror native world Y once; keep a proper rotation for light emission."""
    x, y, z = map(float, position_cm)
    pitch, yaw, roll = map(float, rotation_pitch_yaw_roll)
    if not all(math.isfinite(v) for v in [x, y, z, pitch, yaw, roll]):
        raise ValueError('light pose must be finite')
    return [x, -y, z], [pitch, -yaw, -roll]


def assert_authored_light_positions(expected, actual):
    """One-to-one physical lamp matching; environment light position is immaterial.

    USD radiometric intensities are converted by UE's own USD importer. This
    check never treats the USD input intensity as a lumen value or fits lamps.
    """
    classes = {'SphereLight': {'PointLightComponent', 'SpotLightComponent'},
               'RectLight': {'RectLightComponent'},
               'DistantLight': {'DirectionalLightComponent'},
               'DomeLight': {'SkyLightComponent'}}
    if len(expected) != len(actual):
        raise ValueError(f'authored/native light count differs: {len(expected)} != {len(actual)}')
    candidates = []
    for light in expected:
        if light['type'] not in classes:
            raise ValueError('unsupported original light type: ' + light['type'])
        matches = []
        for j, row in enumerate(actual):
            if row['component_class'] not in classes[light['type']]:
                continue
            position = list(map(float, row['position_ue_cm']))
            if len(position) != 3 or not all(math.isfinite(v) for v in position):
                continue
            error = (math.dist(position, light['expected_native_ue_cm']) / 100
                     if light['type'] not in {'DistantLight', 'DomeLight'} else 0.)
            if error <= .0005:
                matches.append((error, j))
        candidates.append(sorted(matches))
    # Perfect bipartite matching is sufficient: every edge already satisfies
    # the exact native pose tolerance. Keep this pure stdlib because the UE
    # editor's embedded Python does not carry the CPU planner's SciPy/NumPy.
    owners = {}
    def augment(index, seen):
        for error, column in candidates[index]:
            if column in seen:
                continue
            seen.add(column)
            previous = owners.get(column)
            if previous is None or augment(previous, seen):
                owners[column] = index
                return True
        return False
    for i, light in enumerate(expected):
        if not augment(i, set()):
            raise ValueError('native light moved or missing from original design: ' + light['prim_path'])
    assigned = {index: column for column, index in owners.items()}
    rows = []
    for i, light in enumerate(expected):
        row = actual[assigned[i]]
        error = next(error for error, j in candidates[i] if j == assigned[i])
        if row.get('intensity') is None or not math.isfinite(float(row['intensity'])):
            raise ValueError('native light intensity is unmeasured')
        if light['effective_usd_intensity'] > 0 and float(row['intensity']) <= 0:
            raise ValueError('positive original light became zero in native import')
        rows.append({'prim_path': light['prim_path'], 'native_component': row['component'],
                     'native_position_error_m': error,
                     'position_is_physical': light['type'] not in {'DistantLight', 'DomeLight'},
                     'native_intensity': row['intensity'],
                     'intensity_conversion': 'UE 5.5 native USD importer, no guessed lumen scaling'})
    return rows
