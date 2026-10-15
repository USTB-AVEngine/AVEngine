"""CPU-only native floor rasterization and complete static triangle clearance."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import datetime
import json
import math
import os
import subprocess
import time
import numpy as np
from .walkable_map import VERSION, RESOLUTION_M, SPECIES_RADII_M, LAYER_KEYS, WalkableMap, source_identity

# Owner 2026-10-15: this is a permissive construction aid, not pose admission.
# PhysicalFloor's original native gate remains unchanged in roster_motion.
SUPPORT_TOLERANCE_M = .03
MAX_SUPPORT_HEIGHT_JUMP_M = .04
MINIMUM_SUPPORT_FRACTION = .85
GROUND_NORMAL_VERTICAL_COSINE = .9
GROUND_REFERENCE_TOLERANCE_M = .08
NONFLOOR_MARGIN_M = 0.
BODY_BOTTOM_M = .10
BODY_TOP_M = 1.8
ALGORITHM_REVISION = 'permissive-support-3cm-step4cm-85pct-v2'
FOOTPRINT_POLICY = 'independent-radius-coverage-v2'


def stage_path(house):
    data = house['input']
    return Path(data.get('visual_source') or Path(data['scene_directory'])/(data['scan_id']+'.glb'))


def native_static_video(house, runtime):
    """Same exact stage margin/basis as NativeContinuousVideo, without a renderer."""
    from avengine.acoustics.runtime import load_habitat_runtime
    hs, _ = load_habitat_runtime(runtime_mode='current-installed', **runtime['runtime'])
    import magnum as mn
    from habitat_sim.metadata import MetadataMediator
    cfg = hs.SimulatorConfiguration()
    cfg.scene_id = str(stage_path(house))
    cfg.enable_physics = True
    cfg.load_semantic_mesh = False
    cfg.create_renderer = False
    cfg.gpu_device_id = -1
    mediator = MetadataMediator(cfg)
    manager = mediator.stage_template_manager
    attributes = manager.create_template(cfg.scene_id, True)
    attributes.margin = 0.
    manager.register_template(attributes, cfg.scene_id)
    configuration = hs.Configuration(cfg, [hs.agent.AgentConfiguration()])
    configuration.metadata_mediator = mediator
    sim = hs.Simulator(configuration)
    if not sim.pathfinder.load_nav_mesh(house['input']['navmesh_source']):
        sim.close()
        raise ValueError('declared original NavMesh failed to load')
    return SimpleNamespace(sim=sim, hs=hs, mn=mn, pathfinder=sim.pathfinder,
                           motion_house=house, motion_inputs={'phase2': True},
                           receipt={'stage': cfg.scene_id}, actors=[])


def _available_guard():
    import psutil
    if psutil.virtual_memory().available < 30_000_000_000:
        raise RuntimeError('resource_pause: available memory below 30 GB')


def _levels(ray, points):
    """Retain holes as unknown, without changing the existing exact ray routine."""
    values = np.full(len(points), np.nan)
    for start in range(0, len(points), ray.MAX_QUERY_POINTS):
        batch = points[start:start+ray.MAX_QUERY_POINTS]
        try:
            values[start:start+len(batch)] = ray.many(batch)
        except ValueError:
            for k, p in enumerate(batch):
                try:
                    values[start+k] = ray.many([p])[0]
                except ValueError:
                    pass
    return values


def _clip_height(triangles, lower, upper):
    """Clip original faces to the body slab before projecting their footprint.

    Keep the low floor portion of a slanted face out of the obstacle footprint.
    All retained polygons still come from original unmodified stage triangles.
    """
    full = (triangles[:, :, 1].min(1) >= lower) & (triangles[:, :, 1].max(1) <= upper)
    polygons = [face[:, [0, 2]] for face in triangles[full]]
    for face in triangles[~full]:
        points = list(face)
        for height, above in ((lower, True), (upper, False)):
            clipped = []
            if not points:break
            previous = points[-1]
            previous_inside = previous[1] >= height if above else previous[1] <= height
            for current in points:
                inside = current[1] >= height if above else current[1] <= height
                if inside != previous_inside:
                    ratio = (height-previous[1])/(current[1]-previous[1])
                    clipped.append(previous+ratio*(current-previous))
                if inside:clipped.append(current)
                previous, previous_inside = current, inside
            points = clipped
        if points:polygons.append(np.asarray(points)[:, [0, 2]])
    return polygons


def projected_clearance(vertices, faces, reference, bounds, points):
    """Original wall/furniture/step clearance above 10 cm, without scan-floor noise.

    Floor faces within 8cm and normal cosine >.9 are traversable. Only faces
    having a vertex strictly above 10cm can block the construction map. Clip to
    that height slab so one high corner cannot project an entire floor facet.
    """
    import shapely
    from shapely.strtree import STRtree
    candidates = []
    total = removed = 0
    x0, z0, x1, z1 = bounds
    padding = max(SPECIES_RADII_M.values()) + RESOLUTION_M
    for start in range(0, len(faces), 50000):
        tri = vertices[faces[start:start+50000]]
        lo, hi = tri.min(1), tri.max(1)
        normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        norm = np.linalg.norm(normal, axis=1)
        ground = ((np.abs(normal[:, 1]) > GROUND_NORMAL_VERTICAL_COSINE*np.maximum(norm, 1e-20))
                  & (np.max(np.abs(tri[:, :, 1]-reference), axis=1) <= GROUND_REFERENCE_TOLERANCE_M))
        removed += int(ground.sum())
        keep = (~ground & (hi[:, 1] > reference+BODY_BOTTOM_M)
                & (lo[:, 1] <= reference+BODY_TOP_M)
                & (hi[:, 0] >= x0-padding) & (lo[:, 0] <= x1+padding)
                & (hi[:, 2] >= z0-padding) & (lo[:, 2] <= z1+padding))
        selected = tri[keep]
        total += len(selected)
        if len(selected):
            polygons = _clip_height(selected, reference+BODY_BOTTOM_M, reference+BODY_TOP_M)
            if polygons:
                # Vectorize equal-size faces; keep vertical/degenerate hulls too.
                for size in {len(poly) for poly in polygons}:
                    coordinates = np.asarray([poly for poly in polygons if len(poly) == size])
                    candidates.append(shapely.convex_hull(shapely.multipoints(coordinates)))
    distance = np.full(len(points), np.inf)
    if candidates and len(points):
        geometries = np.concatenate(candidates)
        tree = STRtree(geometries)
        max_radius = max(SPECIES_RADII_M.values())
        for start in range(0, len(points), 2048):
            p = shapely.points(points[start:start+2048])
            pairs = tree.query(p, predicate='dwithin', distance=max_radius+RESOLUTION_M+1e-9)
            if pairs.size:
                pi, ti = pairs
                np.minimum.at(distance, start+pi, shapely.distance(p[pi], geometries[ti]))
    return distance, {'selected_original_nonfloor_triangles': total, 'removed_original_ground_triangles': removed}


def native_support_readback(video, points):
    """Use the same native stage rays, with the map's separate 3cm proposal band."""
    result = np.full(len(points), np.nan)
    for index, point in enumerate(points):
        hits = video.sim.cast_ray(video.hs.geo.Ray(video.mn.Vector3(point+[0., .2, 0.]),
                                  video.mn.Vector3(0., -1., 0.)), .85).hits
        if hits:result[index] = float(hits[0].point[1])
    return result


def support_fraction(mask, radius, resolution=RESOLUTION_M):
    """Fraction of actual-radius disk cells supported, counting outside as unknown."""
    from scipy.ndimage import convolve
    n = math.ceil(radius/resolution)
    zz, xx = np.mgrid[-n:n+1, -n:n+1]
    footprint = ((xx*resolution)**2+(zz*resolution)**2 <= radius**2+1e-12).astype(float)
    return convolve(np.asarray(mask, float), footprint, mode='constant', cval=0.)/footprint.sum()


def build_room_map(video, house, room, *, code_commit, output):
    import shapely
    from shapely.geometry import shape
    from scipy.ndimage import maximum_filter, minimum_filter
    from avengine.assets.roster_motion.episode import PhysicalFloor
    from avengine.assets.v7_phase2_floor import original_native_stage_mesh
    began = time.monotonic()
    _available_guard()
    polygon = shape(room['floor_polygon'])
    reference = float(room['measured_floor_y_m'])
    resolution = RESOLUTION_M
    origin = np.floor(np.array(polygon.bounds[:2])/resolution)*resolution-resolution
    xmax, zmax = np.array(polygon.bounds[2:])+resolution
    nx, nz = np.ceil((np.array([xmax, zmax])-origin)/resolution).astype(int)
    gx, gz = np.meshgrid(origin[0]+(np.arange(nx)+.5)*resolution,
                         origin[1]+(np.arange(nz)+.5)*resolution)
    points = np.column_stack((gx.ravel(), gz.ravel()))
    inside = shapely.covers(polygon, shapely.points(points))
    # Entire destination cell must be inside the room polygon.
    half = resolution/2
    for dx, dz in ((-half, -half), (-half, half), (half, -half), (half, half)):
        inside &= shapely.covers(polygon, shapely.points(points+[dx, dz]))
    nav = np.zeros(len(points), bool)
    pf = video.pathfinder
    for k in np.flatnonzero(inside):
        p = np.array([points[k, 0], reference, points[k, 1]])
        snap = np.asarray(pf.snap_point(p), float)
        if np.isfinite(snap).all() and np.linalg.norm((snap-p)[[0, 2]]) <= .002:
            nav[k] = True
    floor = PhysicalFloor(video, reference)
    indices = np.flatnonzero(nav)
    support = np.zeros(len(points), bool)
    floor_y = np.full(len(points), np.nan)
    native_ray_count = 0
    native_errors = []
    if len(indices):
        xyz = np.column_stack((points[indices, 0], np.full(len(indices), reference), points[indices, 1]))
        levels = np.asarray([_levels(floor.original_ray, xyz+[dx, 0., dz])
                             for dx, dz in ((0., 0.), (-half, -half), (-half, half), (half, -half), (half, half))])
        native_y = native_support_readback(video, xyz)
        native_ray_count = len(xyz)
        agreement = np.isfinite(levels[0]) & np.isfinite(native_y)
        native_errors = np.abs(levels[0, agreement]-native_y[agreement]).tolist()
        # The native final PhysicalFloor gate is intentionally not invoked here:
        # it still has its original 15mm bound and would silently undo the 3cm map.
        support[indices] = (np.isfinite(levels).all(0)
                            & (np.max(np.abs(levels-reference), axis=0) <= SUPPORT_TOLERANCE_M)
                            & (np.ptp(levels, axis=0) <= MAX_SUPPORT_HEIGHT_JUMP_M)
                            & np.isfinite(native_y) & (np.abs(native_y-reference) <= SUPPORT_TOLERANCE_M))
        floor_y[indices] = native_y
    shape2 = (nz, nx)
    support2 = support.reshape(shape2)
    # Unknown neighbours do not fabricate a jump back to the reference plane.
    y = floor_y.reshape(shape2)
    high = maximum_filter(np.where(np.isfinite(y), y, -np.inf), size=3, mode='constant', cval=-np.inf)
    low = minimum_filter(np.where(np.isfinite(y), y, np.inf), size=3, mode='constant', cval=np.inf)
    support2 &= (high-low <= MAX_SUPPORT_HEIGHT_JUMP_M)
    mesh, transform = original_native_stage_mesh(video, stage_path(house))
    distances, triangle_stats = projected_clearance(np.asarray(mesh.vertices), np.asarray(mesh.faces), reference,
                                                   polygon.bounds, points)
    layers = {}
    fractions = {}
    for species, radius in SPECIES_RADII_M.items():
        if species == 'slide':
            layers[species] = layers['cat'].copy()
            fractions[species] = fractions['cat']
            continue
        fraction = support_fraction(support2, radius, resolution)
        fractions[species] = fraction
        supported = support2 & (fraction >= MINIMUM_SUPPORT_FRACTION-1e-12)
        clear = distances.reshape(shape2) > radius
        # Native snap is required at the centre. Static geometry already supplies
        # the body radius, so add no second NavMesh/body/cell inflation halo.
        layers[species] = supported & clear
    sources = source_identity([stage_path(house), house['input']['navmesh_source']])
    metadata = {'version': VERSION, 'algorithm_revision': ALGORITHM_REVISION, 'room_id': room['room_label'], 'house_id': house['house'],
                'family': house['family'], 'source_code_commit': code_commit,
                'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'source_identity': sources, 'floor_polygon': room['floor_polygon'],
                'species_radii_m': SPECIES_RADII_M, 'resolution_m': resolution,
                'support_tolerance_m': SUPPORT_TOLERANCE_M, 'native_final_gate_support_tolerance_m': .015,
                'maximum_support_probe_span_m': MAX_SUPPORT_HEIGHT_JUMP_M,
                'maximum_neighbour_height_jump_m': MAX_SUPPORT_HEIGHT_JUMP_M,
                'minimum_footprint_support_fraction': MINIMUM_SUPPORT_FRACTION,
                'footprint_policy': FOOTPRINT_POLICY,
                'native_support_ray_origin_above_reference_m': .2, 'native_support_ray_length_m': .85,
                'nonfloor_margin_m': NONFLOOR_MARGIN_M, 'body_height_slab_m': [BODY_BOTTOM_M, BODY_TOP_M],
                'ground_normal_vertical_cosine': GROUND_NORMAL_VERTICAL_COSINE, 'ground_reference_tolerance_m': GROUND_REFERENCE_TOLERANCE_M,
                'cell_clearance_padding_m': 0., 'extra_navmesh_obstacle_clearance_m': 0.,
                'native_stage_transform': transform, 'native_support_rays': native_ray_count,
                'native_original_ray_max_error_m': max(native_errors, default=None),
                'original_full_pose_gates_unchanged': True, 'renderer_created': False, 'gpu_used': False,
                'qualification': 'permissive construction aid, not a walking admission gate; final native full-pose admission required',
                'cell_counts': {'inside_room': int(inside.sum()), 'native_snap': int(nav.sum()),
                                'supported': int(support2.sum()), **{s: int(a.sum()) for s, a in layers.items()}},
                **triangle_stats, 'compute_seconds': time.monotonic()-began}
    result = WalkableMap(origin, resolution, reference, layers, metadata, floor_y.reshape(shape2))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name+'.'+str(os.getpid())+'.tmp')
    with temporary.open('xb') as stream:
        np.savez_compressed(stream, version=np.array(VERSION), origin_xz_m=origin,
                            resolution_m=np.array(resolution), floor_reference_y_m=np.array(reference),
                            floor_y_m=floor_y.reshape(shape2), source_code_commit=np.array(code_commit),
                            metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
                            inside_room=inside.reshape(shape2), native_snap=nav.reshape(shape2),
                            support=support2, static_clearance_m=distances.reshape(shape2),
                            support_fraction_human=fractions['human'].astype(np.float32),
                            support_fraction_dog=fractions['dog'].astype(np.float32),
                            support_fraction_cat=fractions['cat'].astype(np.float32),
                            **{LAYER_KEYS[s]: a for s, a in layers.items()})
    # Cache files are immutable. Concurrent workers never replace an existing map.
    try:
        os.link(temporary, output)
    finally:
        temporary.unlink()
    output.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    return result


def render_map_png(walkable, mesh, output):
    """Actual original-stage triangle top view, with each species mask overlaid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    floor = walkable.floor_reference
    bounds = (*walkable.origin, *(walkable.origin+np.array(walkable.shape[::-1])*walkable.resolution))
    x0, z0, x1, z1 = bounds
    selected = []
    heights = []
    for start in range(0, len(faces), 50000):
        tri = vertices[faces[start:start+50000]]
        lo, hi = tri.min(1), tri.max(1)
        keep = ((hi[:, 0] >= x0) & (lo[:, 0] <= x1) & (hi[:, 2] >= z0) & (lo[:, 2] <= z1)
                & (hi[:, 1] >= floor-.03) & (lo[:, 1] <= floor+BODY_TOP_M))
        selected.append(tri[keep][:, :, [0, 2]])
        heights.append(tri[keep, :, 1].mean(1))
    tri, y = np.concatenate(selected), np.concatenate(heights)
    order = np.argsort(y)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    for ax, species in zip(axes, ('human', 'dog', 'cat')):
        ax.add_collection(PolyCollection(tri[order], array=y[order]-floor, cmap='Greys', clim=(-.1, 2.5), linewidths=0, rasterized=True))
        overlay = np.zeros((*walkable.shape, 4))
        overlay[:, :, :] = [1., .10, .10, .20]
        overlay[walkable.layers[species]] = [.05, .9, .2, .65]
        ax.imshow(overlay, extent=(x0, x1, z0, z1), origin='lower', interpolation='nearest', zorder=2)
        ax.set(xlim=(x0, x1), ylim=(z0, z1), aspect='equal', xlabel='world X (m)', ylabel='world Z (m)',
               title=f'{species}, radius {SPECIES_RADII_M[species]:.2f} m: {walkable.layers[species].sum()} cells')
    fig.suptitle(walkable.metadata['room_id']+'\nOriginal stage triangle top view; green=walkable, red=blocked; 5 cm cells')
    fig.savefig(output, dpi=130)
    plt.close(fig)
