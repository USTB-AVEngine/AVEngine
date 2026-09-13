"""Build reusable CPU navigation from a room's declared render surface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def stage_for_manifest(room, manifest_path, stage_config=None):
    """Reuse explicit stage orientation or the already-normalized render mesh."""
    if stage_config is not None:
        stage_config = Path(stage_config).resolve()
        stage = json.loads(stage_config.read_text())
        for key in ('render_asset', 'collision_asset', 'semantic_asset'):
            if stage.get(key):
                value = Path(stage[key])
                stage[key] = str(value if value.is_absolute() else (stage_config.parent / value).resolve())
        stage['collision_asset'] = stage['render_asset']
        return stage
    paths = [row['path'] for row in room.get('assets', ())
             if row.get('role') == 'render_surface_mesh']
    if len(paths) != 1:
        raise ValueError('Declare one render_surface_mesh or provide its stage configuration')
    coordinates = room.get('coordinate_system', {})
    if coordinates.get('up_axis') != '+Y' or coordinates.get('linear_unit') != 'meter':
        raise ValueError('A surface without stage orientation must already use shared meter/+Y coordinates')
    surface = Path(paths[0])
    if not surface.is_absolute():
        surface = (Path(manifest_path).parent / surface).resolve()
    if not surface.is_file():
        raise FileNotFoundError(surface)
    return {'render_asset': str(surface), 'collision_asset': str(surface),
            'up': [0, 1, 0], 'front': [0, 0, -1], 'units_to_meters': 1.0}


def build(args):
    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime

    manifest_path = args.room_manifest.resolve()
    room = json.loads(manifest_path.read_text())
    stage = stage_for_manifest(room, manifest_path, args.stage_config)
    navigation = room['navigation']
    height, radius = float(navigation['agent_height_m']), float(navigation['agent_radius_m'])
    if height <= 0 or radius <= 0:
        raise ValueError('Use positive agent dimensions from the room navigation declaration')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stage_path = output / 'render_surface.stage_config.json'
    stage_path.write_text(json.dumps(stage, indent=2) + '\n')
    runtime = prepare_installed_habitat_runtime(
        runtime_prefix=args.runtime_prefix, magnum_python_site=args.magnum_site,
        rlr_sdk_root=args.rlr_sdk_root)
    hs = runtime.habitat_sim
    settings = hs.NavMeshSettings()
    settings.set_defaults()
    settings.agent_height, settings.agent_radius = height, radius
    settings.cell_size, settings.cell_height = .05, .01
    settings.include_static_objects = False  # The declared room surface already contains its furniture.
    cfg = hs.SimulatorConfiguration()
    cfg.scene_id = str(stage_path)
    cfg.physics_config_file = str(runtime.physics_config_path)
    cfg.create_renderer = False
    cfg.requires_textures = False
    cfg.enable_physics = False
    cfg.navmesh_settings = settings
    agent = hs.agent.AgentConfiguration()
    agent.sensor_specifications = []
    started = time.monotonic()
    with hs.Simulator(hs.Configuration(cfg, [agent])) as simulator:
        pf = simulator.pathfinder
        if not simulator.recompute_navmesh(pf, settings) or not pf.is_loaded:
            raise RuntimeError('The declared room surface did not produce navigation')
        navmesh = output / 'navigation.navmesh'
        pf.save_nav_mesh(str(navmesh))
        report = {
            'status': 'built', 'renderer': 'disabled', 'room_manifest': str(manifest_path),
            'room_id': room.get('room_id'), 'surface': stage, 'navmesh': str(navmesh),
            'settings': {key: getattr(settings, key) for key in (
                'agent_height', 'agent_radius', 'cell_size', 'cell_height',
                'agent_max_climb', 'agent_max_slope')},
            'navigable_area_m2': float(pf.navigable_area), 'num_islands': int(pf.num_islands),
            'elapsed_s': time.monotonic() - started,
            'claim_boundary': 'Reusable navigation from the room surface. Native episode results decide QA acceptance.',
        }
    (output / 'build_result.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--room-manifest', required=True, type=Path)
    parser.add_argument('--stage-config', type=Path, help='Existing stage orientation for a scan or other native surface')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--runtime-prefix', required=True, type=Path)
    parser.add_argument('--magnum-site', required=True, type=Path)
    parser.add_argument('--rlr-sdk-root', required=True, type=Path)
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == '__main__':
    main()
