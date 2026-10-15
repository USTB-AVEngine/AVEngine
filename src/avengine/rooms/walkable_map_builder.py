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

# PhysicalFloor admits 15 mm. The requested cache conservatively narrows it to 10 mm.
SUPPORT_TOLERANCE_M = .01
NATIVE_SUPPORT_TOLERANCE_M = .015
NONFLOOR_MARGIN_M = .0200001
BODY_BOTTOM_M = .00101  # also cover obstacles below the requested 5 cm ray band
BODY_TOP_M = 1.8


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


def projected_clearance(vertices, faces, reference, bounds, points):
    """Distance to projection of EVERY nonfloor triangle in the body height slab.

    A projected triangle is a superset of its slab-clipped projection. Thus this
    can reject extra points but cannot miss an intermediate-height triangle as
    a finite collection of horizontal rays could. The ground classification is
    exactly TriangleClearance's .98 normal / 2 cm reference-plane test.
    """
    import shapely
    from shapely.strtree import STRtree
    candidates = []
    total = removed = 0
    x0, z0, x1, z1 = bounds
    padding = max(SPECIES_RADII_M.values()) + NONFLOOR_MARGIN_M + RESOLUTION_M
    for start in range(0, len(faces), 50000):
        tri = vertices[faces[start:start+50000]]
        lo, hi = tri.min(1), tri.max(1)
        normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        norm = np.linalg.norm(normal, axis=1)
        ground = ((np.abs(normal[:, 1]) > .98*np.maximum(norm, 1e-20))
                  & (np.max(np.abs(tri[:, :, 1]-reference), axis=1) < .02))
        removed += int(ground.sum())
        keep = (~ground & (hi[:, 1] >= reference+BODY_BOTTOM_M)
                & (lo[:, 1] <= reference+BODY_TOP_M)
                & (hi[:, 0] >= x0-padding) & (lo[:, 0] <= x1+padding)
                & (hi[:, 2] >= z0-padding) & (lo[:, 2] <= z1+padding))
        selected = tri[keep][:, :, [0, 2]]
        if len(selected):
            # Convex hull preserves vertical triangles as lines, rather than
            # dropping their degenerate 2D polygons and losing thin walls.
            candidates.append(shapely.convex_hull(shapely.multipoints(selected)))
            total += len(selected)
    distance = np.full(len(points), np.inf)
    if candidates and len(points):
        geometries = np.concatenate(candidates)
        tree = STRtree(geometries)
        max_radius = max(SPECIES_RADII_M.values())+NONFLOOR_MARGIN_M+RESOLUTION_M/math.sqrt(2)
        for start in range(0, len(points), 2048):
            p = shapely.points(points[start:start+2048])
            pairs = tree.query(p, predicate='dwithin', distance=max_radius+1e-9)
            if pairs.size:
                pi, ti = pairs
                distances = shapely.distance(p[pi], geometries[ti])
                np.minimum.at(distance, start+pi, distances)
    return distance, {'selected_original_nonfloor_triangles': total, 'removed_original_ground_triangles': removed}


def build_room_map(video, house, room, *, code_commit, output):
    import shapely
    from shapely.geometry import shape
    from scipy.ndimage import minimum_filter
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
    nav_clearance = np.zeros(len(points), float)
    nav_y = np.full(len(points), np.nan)
    pf = video.pathfinder
    for k in np.flatnonzero(inside):
        p = np.array([points[k, 0], reference, points[k, 1]])
        snap = np.asarray(pf.snap_point(p), float)
        if np.isfinite(snap).all() and np.linalg.norm((snap-p)[[0, 2]]) <= .002:
            nav[k] = True
            nav_y[k] = snap[1]
            nav_clearance[k] = float(pf.distance_to_closest_obstacle(snap, 1.)) if hasattr(pf, 'distance_to_closest_obstacle') else np.inf
    floor = PhysicalFloor(video, reference)
    indices = np.flatnonzero(nav)
    support = np.zeros(len(points), bool)
    floor_y = np.full(len(points), np.nan)
    local_span = np.zeros(len(points))
    if len(indices):
        xyz = np.column_stack((points[indices, 0], np.full(len(indices), reference), points[indices, 1]))
        levels = []
        for dx, dz in ((0., 0.), (-half, -half), (-half, half), (half, -half), (half, half)):
            levels.append(_levels(floor.original_ray, xyz+[dx, 0., dz]))
        levels = np.asarray(levels)
        support[indices] = (np.isfinite(levels).all(0) & (np.max(np.abs(levels-reference), axis=0) <= SUPPORT_TOLERANCE_M)
                            & (np.ptp(levels, axis=0) <= NATIVE_SUPPORT_TOLERANCE_M)
                            & (np.abs(nav_y[indices]-reference) <= NATIVE_SUPPORT_TOLERANCE_M))
        floor_y[indices] = levels[0]
        local_span[indices] = np.ptp(levels, axis=0)
        # Read back every initially supported cell independently from the native
        # collision world through PhysicalFloor.__call__ (including .1mm agreement).
        for k in indices[support[indices]]:
            try:
                y = floor(np.array([points[k, 0], reference, points[k, 1]]))
                support[k] = abs(y-reference) <= SUPPORT_TOLERANCE_M
            except ValueError:
                support[k] = False
    shape2 = (nz, nx)
    support2 = support.reshape(shape2)
    # Height discontinuities at neighbouring cell probes cannot exceed the
    # existing PhysicalFloor 15mm bound. Unknown neighbours remain blocked.
    finite_y = np.where(np.isfinite(floor_y.reshape(shape2)), floor_y.reshape(shape2), reference)
    high = -minimum_filter(-finite_y, size=3, mode='constant', cval=-reference)
    low = minimum_filter(finite_y, size=3, mode='constant', cval=reference)
    support2 &= (high-low <= NATIVE_SUPPORT_TOLERANCE_M)
    mesh, transform = original_native_stage_mesh(video, stage_path(house))
    distances, triangle_stats = projected_clearance(np.asarray(mesh.vertices), np.asarray(mesh.faces), reference,
                                                   polygon.bounds, points)
    layers = {}
    # Square support probes and complete footprint erosion reject furniture,
    # drop-offs and discontinuities underneath any part of the body footprint.
    for species, radius in SPECIES_RADII_M.items():
        if species == 'slide':
            layers[species] = layers['cat'].copy()
            continue
        reach = radius+resolution/math.sqrt(2)
        n = math.ceil(reach/resolution)
        zz, xx = np.mgrid[-n:n+1, -n:n+1]
        footprint = (xx*resolution)**2+(zz*resolution)**2 <= (reach+resolution/math.sqrt(2))**2
        from scipy.ndimage import binary_erosion
        supported = binary_erosion(support2, structure=footprint, border_value=0)
        clear = distances.reshape(shape2) > radius+NONFLOOR_MARGIN_M+resolution/math.sqrt(2)
        layers[species] = supported & clear & (nav_clearance.reshape(shape2) >= radius+resolution/math.sqrt(2))
    sources = source_identity([stage_path(house), house['input']['navmesh_source']])
    metadata = {'version': VERSION, 'room_id': room['room_label'], 'house_id': house['house'],
                'family': house['family'], 'source_code_commit': code_commit,
                'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'source_identity': sources, 'floor_polygon': room['floor_polygon'],
                'species_radii_m': SPECIES_RADII_M, 'resolution_m': resolution,
                'support_tolerance_m': SUPPORT_TOLERANCE_M, 'native_support_tolerance_m': NATIVE_SUPPORT_TOLERANCE_M,
                'maximum_support_probe_span_m': NATIVE_SUPPORT_TOLERANCE_M,
                'native_support_ray_origin_above_reference_m': .2, 'native_support_ray_length_m': .85,
                'nonfloor_margin_m': NONFLOOR_MARGIN_M, 'body_height_slab_m': [BODY_BOTTOM_M, BODY_TOP_M],
                'ground_normal_vertical_cosine': .98, 'ground_reference_tolerance_m': .02,
                'cell_clearance_padding_m': resolution/math.sqrt(2),
                'native_stage_transform': transform, 'native_support_rays': floor.native_ray_count,
                'native_original_ray_max_error_m': max(floor.native_ray_agreements, default=None),
                'original_full_pose_gates_unchanged': True, 'renderer_created': False, 'gpu_used': False,
                'qualification': 'conservative static cylinder proposal filter; final native full-pose admission required',
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
