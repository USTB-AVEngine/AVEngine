"""Explicit production versions and frozen CPU poses; no renderer is imported here."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def parent_hash(house, geometry):
    value = {'house': house, 'camera_m': geometry['camera_m'],
             'feet_m': sorted(geometry['feet_m']), 'basis': geometry['basis']}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def validate_versions(config, *, require_host=True):
    versions = config.get('production_versions')
    if not versions:
        if config.get('split') == 'test':
            raise ValueError('formal production is pending three accepted version deliveries')
        return None
    for name in ('generator', 'assets', 'render_environment'):
        row = versions[name]
        if row.get('status') != 'accepted':
            raise ValueError('production delivery is pending: ' + name)
        evidence = row['acceptance_report']
        if digest(evidence) != row['acceptance_report_sha256']:
            raise ValueError('acceptance evidence identity changed: ' + name)
        if not row.get('revision'):
            raise ValueError('missing version identity: ' + name)
        for path, expected in row.get('files_sha256', {}).items():
            if digest(path) != expected:
                raise ValueError('production dependency identity changed: ' + path)
    generator = versions['generator']
    repository = Path(generator['repo']).resolve()
    head = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repository), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if dirty:
        raise ValueError('accepted generator checkout has tracked modifications')
    if head != generator['revision']:
        raise ValueError('generator commit differs from the accepted version')
    if generator['stage_contract'] != 'frozen_panel_stages_v1':
        raise ValueError('generator needs prepare/capture/audio --config --index adapter')
    entrypoint = (repository / generator['entrypoint']).resolve()
    if not entrypoint.is_relative_to(repository) or digest(entrypoint) != generator['entrypoint_sha256']:
        raise ValueError('generator entrypoint differs from the accepted version')
    assets = versions['assets']
    for field in ('registry', 'template_plan'):
        if str(Path(config[field]).resolve()) != str(Path(assets[field]).resolve()):
            raise ValueError('config/asset version mismatch: ' + field)
        if digest(config[field]) != assets[field + '_sha256']:
            raise ValueError('asset input changed: ' + field)
    actors = read(config['template_plan'])['visual_plan']['actors']
    actual = [{'asset_id': a['asset_id'], 'asset_revision': a['asset_revision']} for a in actors]
    if actual != assets['actor_assets']:
        raise ValueError('template actor assets/revisions differ from the accepted fixed pair')
    environment = versions['render_environment']
    if config['runtime'] != environment['runtime']:
        raise ValueError('config/render runtime mismatch')
    if require_host and socket.gethostname() not in environment['hostnames']:
        raise ValueError('production host has not been accepted for this render environment')
    if config.get('split') == 'test' and environment.get('machine') != 'CCI':
        raise ValueError('formal data production is restricted to CCI')
    return str(entrypoint)


def prepared_parent(config, index):
    """Resolve one reserved parent without changing the frozen seed or quota."""
    source = config.get('prepared_geometry_root')
    if not source:
        return None
    per = int(config['groups_per_house'])
    hi, ordinal = divmod(index, per)
    house = config['houses'][hi]
    prepared = read(Path(source) / house['house'] / 'prepared.json')
    row = next(g for g in prepared['groups'] if g['ordinal'] == ordinal)
    cell = (ordinal + 2 * hi) % 4
    expected = {'voice': 'same' if cell < 2 else 'different',
                'side': 'opposite' if cell % 2 == 0 else 'same_side'}
    if (row['status'] != 'cpu_preflight_pass' or row['house'] != house['house'] or
            row['quota'] != expected or row.get('answers_generated') is not False):
        raise ValueError('prepared house/quota/split does not match this production parent')
    if row['room'] not in {r['room_label'] for r in house['rooms']}:
        raise ValueError('prepared room is outside the accepted scope')
    key = parent_hash(house['house'], row['geometry'])
    if key != row['physical_parent_sha256'] or key in config.get('excluded_physical_parent_sha256', []):
        raise ValueError('prepared parent hash changed or reuses a pilot parent')
    if config.get('split') == 'test':
        seed = next(g for g in read(config['frozen_seed_plan'])['groups']
                    if g['house'] == house['house'] and int(g['group_id'].rsplit('_', 1)[1]) == ordinal)
        if row['scene_seed'] != seed['independent_scene_seed'] or row['frozen_group_id'] != seed['group_id']:
            raise ValueError('production parent no longer uses the frozen seed table')
    if config.get('prepared_geometry_sha256') and digest(Path(source) / house['house'] / 'prepared.json') != config['prepared_geometry_sha256'][house['house']]:
        raise ValueError('prepared geometry changed after version binding')
    return row


def versions_identity(versions):
    return hashlib.sha256(json.dumps(versions, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def require_canary_acceptance(config):
    """Formal production must use the exact versions that passed a twenty-group canary."""
    if config.get('split') != 'test':
        return
    receipt = config.get('canary_acceptance')
    if not receipt:
        raise ValueError('formal production requires the new-version twenty-group canary')
    expected = versions_identity(config['production_versions'])
    if receipt['production_versions_sha256'] != expected:
        raise ValueError('formal versions differ from the admitted canary')
    values = {}
    for key in ('comparison', 'verification', 'sizes'):
        item = receipt[key]
        if digest(item['path']) != item['sha256']:
            raise ValueError('canary evidence commitment changed: ' + key)
        values[key] = read(item['path'])
        if values[key]['status'] != 'pass':
            raise ValueError('canary evidence is not passing: ' + key)
    comparison = values['comparison']
    verification = values['verification']
    if (comparison.get('production_versions_sha256_after') != expected or
            len(comparison['groups']) != 20 or not all(g['status'] == 'pass' for g in comparison['groups']) or
            len(verification['groups']) != 20 or not all(g['three_layers_verified'] for g in verification['groups']) or
            verification.get('scoring_script_audit', {}).get('status') != 'pass' or
            verification.get('external_dependency_reads_resolved_to_retained_files') is not True or
            verification.get('retained_configuration_commitment_verified') is not True or
            values['sizes']['groups'] != 20 or values['sizes']['temporary_remaining']['allocated_bytes'] != 0):
        raise ValueError('incomplete new-version twenty-group canary admission')
