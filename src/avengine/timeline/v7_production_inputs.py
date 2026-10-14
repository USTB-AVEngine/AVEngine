"""Read-only v7 production input contract, before native placement and QA.

Room witnesses restrict candidate categories; they never certify a combined
parent, override historical acoustic gates, or bypass the original validators.
"""
from __future__ import annotations
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import random

SCHEMA = 'v7-production-inputs-v1'
COLORS = ('black', 'blue', 'green', 'white', 'yellow')


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint(path):
    p = Path(path)
    digest = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest.hexdigest()}


def prefer_v2(path):
    p = Path(path)
    candidate = p.with_name(p.stem + '_v2' + p.suffix)
    return candidate if candidate.is_file() else p


def _cell_usable(cell, family):
    if cell.get('budget_limited') or cell.get('availability_kind') == 'budget_limited':
        return False
    if any(str(cell.get(k, '')).startswith('budget_limited') for k in ('status', 'availability')):
        return False
    if family == 'hm3d':
        return (cell.get('usable') is True and
                cell.get('sampler_usable_with_original_house_acoustic_gate') is True)
    if family == 'mp3d':
        return cell.get('available') is True
    return cell.get('status') == 'usable_for_tested_representative'


def normalize_rooms(document, family, test_houses):
    raw = document['rooms']
    entries = raw.values() if isinstance(raw, dict) else raw
    rooms = []
    for r in entries:
        house = r['house']
        if house in test_houses or r.get('split', r.get('task_split', 'train')) != 'train':
            continue
        if r.get('room_removed') or r.get('physical_unusability_confirmed'):
            continue
        if family == 'mp3d' and (r.get('first_layer_available') is not True or
                                r.get('production_sampling_available') is not True):
            continue
        if family == 'kujiale' and (r.get('native_room_render_status') != 'pass' or
                                   r.get('floor_support_check', {}).get('status') != 'pass'):
            continue
        cells = r['categories']
        pairs = cells.items() if isinstance(cells, dict) else ((c['category_key'], c) for c in cells)
        allowed, witnesses = [], {}
        for category, cell in pairs:
            if category == 'floor_animals' or not _cell_usable(cell, family):
                continue
            category = 'chair_seated' if category == 'chair_seated_human' else category
            allowed.append(category)
            witnesses[category] = cell.get('representative_asset_id', cell.get('tested_asset_id', cell.get('witness_asset_id')))
        if not allowed:
            continue
        rooms.append({'family': family, 'house': house, 'room_id': r['room_id'],
                      'key': house + '::' + r['room_id'], 'categories': sorted(set(allowed)),
                      'witness_asset_ids': witnesses,
                      'listener_m': r.get('listener_m'),
                      'historical_house_acoustic_status': r.get('historical_house_acoustic_status', r.get('original_house_acoustic_gate')),
                      'original_combined_parent_gates_required': True})
    if len({r['key'] for r in rooms}) != len(rooms):
        raise ValueError('duplicate house/room legality key')
    return rooms


def asset_category(asset, request, family):
    kind = asset.get('kind', asset.get('capability', {}).get('asset_kind'))
    species = asset.get('species_id')
    if kind in {'human', 'performer', 'standing_human'} or species == 'human':
        if family == 'kujiale':
            return 'human.seated' if request.get('action') == 'sit' else 'human.standing'
        return 'chair_seated' if request.get('action') == 'sit' else 'standing_human'
    if kind == 'chair':
        return 'chair_seated' if family != 'kujiale' else 'human.seated'
    if species in {'dog', 'cat', 'bird'} or kind in {'dog', 'cat', 'bird'}:
        species = species or kind
        return ('animal.' + species) if family == 'kujiale' else ('perched_bird' if species == 'bird' else 'floor_' + species)
    surface = asset.get('attachment_surface')
    if surface not in {'floor', 'table', 'wall', 'ceiling'}:
        raise ValueError('asset lacks a declared attachment category: ' + asset['asset_id'])
    if family == 'kujiale':
        object_type = asset.get('object_type')
        if not object_type:
            raise ValueError('UE object lacks its detailed room category')
        return 'object.' + object_type + '.' + surface
    return surface + '_object'


def validate_recipe(row, catalog, manifest):
    if manifest.get('schema') != SCHEMA:
        raise ValueError('production input contract required')
    if row.get('split') != 'train' or row['house'] in manifest['test_houses']:
        raise ValueError('test house or non-training recipe excluded')
    key = row['house'] + '::' + row['room_id']
    room = next((r for r in manifest['rooms'] if r['key'] == key), None)
    if room is None:
        raise ValueError('room not eligible under the original acoustic and legality gates: ' + key)
    assets = {a['asset_id']: a for a in catalog}
    requests = row.get('motion_requests', {})
    if room['family'] == 'kujiale' and requests:
        raise ValueError('Kujiale registered producer does not consume phase2 motion requests')
    colors = []
    categories = {}
    if len(row['asset_ids']) != len(set(row['asset_ids'])):
        raise ValueError('duplicate recipe body')
    if not set(requests) <= set(row['asset_ids']):
        raise ValueError('motion request is not bound to a recipe body')
    for aid in row['asset_ids']:
        if aid in manifest.get('policy', {}).get('temporarily_excluded_asset_ids', {}):
            raise ValueError('temporarily excluded source asset: ' + aid)
        a = assets[aid]
        category = asset_category(a, requests.get(aid, {}), room['family'])
        if category not in room['categories']:
            raise ValueError('room does not admit asset category: ' + aid + '/' + category)
        categories[aid] = category
        if a.get('controlled_shirt'):
            colors.append(a.get('color', a.get('registered_label_color')))
        if a.get('species_id') in {'dog', 'cat'} and requests.get(aid, {}).get('action') == 'walk':
            authority = manifest['animal_motion'].get(aid)
            if authority is None or requests[aid].get('animal_motion_admission') != authority:
                raise ValueError('animal motion is not bound to its final admission')
    if any(c not in COLORS for c in colors) or len(colors) != len(set(colors)):
        raise ValueError('controlled shirts must use distinct admitted non-red colors')
    for aid, req in requests.items():
        if req.get('action') == 'sit':
            chair = req.get('chair_id')
            if chair and (chair not in assets or asset_category(assets[chair], {}, room['family']) not in room['categories']):
                raise ValueError('fallback chair not admitted in the room')
    return {'status': 'pass', 'room_key': key, 'categories': categories,
            'controlled_shirt_colors': colors, 'test_houses_excluded': True,
            'combined_native_parent_admitted': False}


@lru_cache(maxsize=8)
def _cached_contract(path, expected_sha256):
    if fingerprint(path)['sha256'] != expected_sha256:
        raise ValueError('production input snapshot hash changed')
    return read_json(path)


def validate_sampling_row(row, catalog):
    receipt = row.get('production_sampling')
    if receipt is None:
        return None  # Legacy private integration recipes retain their own gates.
    contract = _cached_contract(receipt['manifest_path'], receipt['manifest_sha256'])
    return validate_recipe(row, catalog, contract)


def animal_authorities(document, source_path):
    result = {}
    for a in document['assets']:
        mode = a['motion_mode']
        if mode not in {'walk_animation', 'slide_no_animation'}:
            raise ValueError('unsupported final animal motion mode')
        selected = a['selected']
        if selected.get('status') != 'pass':
            raise ValueError('unadmitted selected animal motion')
        result[a['asset_id']] = {'admission_path': str(source_path), 'asset_id': a['asset_id'],
                                 'mode': mode, 'native_motion': selected['native_motion'],
                                 'source_master': selected['source_master'],
                                 'motion_marker': 'animated_walk' if mode == 'walk_animation' else 'intentional_slide_no_animation',
                                 'audio_admission_replaced': False}
        for key in ('native_motion_sha256', 'source_master_sha256', 'pose_reconstruction_required'):
            if key in selected:
                result[a['asset_id']][key] = selected[key]
    return result


def sample_unique_shirts(catalog, count, seed):
    if not 0 <= count <= len(COLORS):
        raise ValueError('at most five controlled shirt colors per scene')
    rng = random.Random(seed)
    colors = rng.sample(list(COLORS), count)
    choices = []
    for color in colors:
        pool = sorted((a for a in catalog if a.get('controlled_shirt') and a.get('color') == color), key=lambda a: a['asset_id'])
        if not pool:
            raise ValueError('missing admitted shirt color: ' + color)
        choices.append(deepcopy(rng.choice(pool)))
    return choices


def sample_dialogue(manifest, scenario, genders, seed):
    rng = random.Random(seed)
    sounds = [r for r in manifest['sounds'] if r['scenario_id'] == scenario]
    roles = sorted({r['role_id'] for r in sounds})
    if len(roles) != len(genders):
        raise ValueError('authored scenario role count differs from actual speakers')
    chosen = []
    for role, gender in zip(roles, genders):
        pool = sorted((r for r in sounds if r['role_id'] == role and r['voice_gender'].lower() in {gender, gender[:1]}), key=lambda r: r['sound_asset_id'])
        if not pool:
            raise ValueError('authored voice pool lacks the registered speaker gender')
        chosen.append(deepcopy(rng.choice(pool)))
    return chosen


def prepare_contract(config):
    split = read_json(config['room_split_summary'])
    tests = split['test_houses']
    if isinstance(tests, dict):
        tests = list(tests)
    tests = [r['house'] if isinstance(r, dict) else r for r in tests]
    contract = {'schema': SCHEMA, 'test_houses': sorted(tests), 'rooms': [], 'inputs': {},
                'policy': {'budget_limited_excluded': True, 'prefer_v2': True,
                           'native_gates_changed': False, 'training_export': False,
                           'shirt_colors': list(COLORS), 'animal_animation_first': True}}
    contract['policy']['temporarily_excluded_asset_ids'] = deepcopy(config.get('temporarily_excluded_asset_ids', {}))
    contract['inputs']['room_split_summary'] = fingerprint(config['room_split_summary'])
    for family, path in config['room_legality'].items():
        path = prefer_v2(path)
        document = read_json(path)
        contract['inputs']['room_' + family] = {**fingerprint(path), 'round2': document.get('round2'),
                                               'summary': document.get('summary')}
        contract['rooms'].extend(normalize_rooms(document, family, set(tests)))
    for key in ('source_admission', 'semantic_manifest', 'animal_admission', 'class_merge_map'):
        contract['inputs'][key] = fingerprint(config[key])
    source = read_json(config['source_admission'])
    from avengine.timeline.v7_visible_sources import validate_english_source_admission, production_audio_record_views
    from avengine.timeline.v7_audio_sources import filter_admission
    language = validate_english_source_admission(source)
    accepted, rejected = filter_admission(production_audio_record_views(source['records']))
    if rejected:
        raise ValueError('v13 source admission rejected by unchanged input gates: ' + str(rejected[:4]))
    classes = sorted({r['sound_class'] for r in accepted})
    if 'telephone' in classes or 'phone_ring' in classes or 'telephone_bell_ringing' not in classes:
        raise ValueError('v13 merged telephone domain required')
    contract['audio'] = {'record_count': len(accepted), 'classes': classes, 'class_count': len(classes), 'language_audit': language}
    semantic = read_json(config['semantic_manifest'])
    voices = sorted({r['voice_reference']['speaker_id'] for r in semantic['sounds']})
    contract['dialogue'] = {'sound_count': len(semantic['sounds']), 'voice_count': len(voices), 'voices': voices}
    contract['animal_motion'] = animal_authorities(read_json(config['animal_admission']), config['animal_admission'])
    return contract


def sample_room_assets(catalog, room, seed, controlled_people=3, object_count=2, animal_count=1, decor_count=0):
    """Default category-constrained candidate mix; native admission is separate.

    A room may have fewer candidate kinds. This never widens its allowed
    categories to satisfy a requested mixture, and never labels a slide gait.
    """
    rng = random.Random(seed)
    allowed = set(room['categories'])
    eligible = [a for a in catalog if asset_category(a, {}, room['family']) in allowed and a.get('kind') != 'chair']
    humans = [a for a in eligible if a.get('controlled_shirt')]
    result = sample_unique_shirts(humans, min(controlled_people, len({a['color'] for a in humans})), seed) if humans else []
    for kind, count in [('object', object_count), ('animal', animal_count)]:
        pool = sorted((a for a in eligible if a.get('kind') == kind and a.get('allowed_classes') and not a.get('silent_decor')), key=lambda a: a['asset_id'])
        result.extend(deepcopy(rng.sample(pool, min(count, len(pool)))))
    silent = sorted((a for a in eligible if a.get('kind') == 'object' and a.get('silent_decor')), key=lambda a:a['asset_id'])
    result.extend(deepcopy(rng.sample(silent, min(decor_count,len(silent)))))
    return result


def bind_animal_motion(request, asset_id, manifest):
    authority = deepcopy(manifest['animal_motion'][asset_id])
    return {**deepcopy(request), 'action': 'walk', 'animal_motion_admission': authority,
            'joint_animation_present': authority['mode'] == 'walk_animation',
            'intentional_slide': authority['mode'] == 'slide_no_animation'}


def validate_registered_sampling_row(row, selected_assets, inputs):
    receipt = row.get('production_sampling')
    if receipt is None:
        return None
    contract = _cached_contract(receipt['manifest_path'], receipt['manifest_sha256'])
    catalog = deepcopy(selected_assets)
    for a in catalog:
        a.setdefault('kind', 'human')
        a.setdefault('controlled_shirt', a.get('color') in COLORS)
    plan = row['production_scenario']
    bindings = plan.get('native_environment', [])
    if bindings:
        native = read_json(inputs['kujiale_ue_asset_admission'])
        records = native['assets']
        records = list(records.values()) if isinstance(records, dict) else records
        by_id = {r['asset_id']: r for r in records}
        for binding in bindings:
            a = deepcopy(by_id[binding['asset_id']])
            if a.get('category') in {'dog', 'cat', 'bird'}:a.setdefault('species_id',a['category'])
            a.setdefault('kind', 'animal' if a.get('species_id') in {'dog', 'cat', 'bird'} else 'object')
            catalog.append(a)
    proxy = {'house': row['house'], 'room_id': row['room_id'], 'split': row['split'],
             'asset_ids': [a['asset_id'] for a in catalog],
             'motion_requests': deepcopy(row.get('motion_requests', {}))}
    return validate_recipe(proxy, catalog, contract)


def attach_runtime_contract(row, inputs):
    """Make an explicit production runtime contract effective for every row."""
    path = inputs.get('production_input_contract')
    if not path:
        return row
    proof = fingerprint(path)
    receipt = {'manifest_path': proof['path'], 'manifest_sha256': proof['sha256']}
    if row.get('production_sampling') is not None and row['production_sampling'] != receipt:
        raise ValueError('recipe and runtime production input contracts differ')
    row = deepcopy(row)
    row['production_sampling'] = receipt
    return row


def bind_semantic_registry(manifest, assets):
    """Bind missing headers to retained human and authored-scenario authorities.

    Every sound record remains exact. Missing question/domain definitions come
    only from the expansion manifest's own SHA-pinned original authored batch.
    Native SHA/frame checks and semantic question gates remain unchanged.
    """
    if manifest is None or (manifest.get('source_registry') and manifest.get('dialogues')):
        return manifest
    result = deepcopy(manifest)
    if not result.get('source_registry'):
        registries = {str(a['source_registry']) for a in assets
                      if a.get('source_registry') and a.get('dialogue_voice_binding_eligible') is not False
                      and (a.get('species_id') == 'human' or a.get('kind') == 'human'
                           or a.get('entity_class') == 'articulated_human')}
        if len(registries) != 1:
            raise ValueError('semantic header requires one actual admitted human registry')
        result['source_registry'] = next(iter(registries))
    proof = {'source': 'actual selected admitted human assets', 'sound_records_changed': False,
             'native_registry_SHA_and_all_frame_qualification_required': True}
    if not result.get('dialogues'):
        lineage = result.get('batch_summary', {})
        path = lineage.get('source_old_manifest')
        expected = lineage.get('source_old_manifest_sha256')
        if not path or not expected:
            raise ValueError('expanded dialogue catalog lacks its pinned original authored authority')
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('original authored dialogue manifest SHA changed')
        original = json.loads(raw)
        dialogues = original['dialogues']
        roles = {(sc['id'], r['role_id']): r for sc in dialogues['scenarios'] for r in sc['roles']}
        for sound in result['sounds']:
            role = roles.get((sound['scenario_id'], sound['role_id']))
            if (role is None or role['transcript'] != sound['transcript']
                    or role['meaning_answer'] != sound['semantic_answer']['answer']):
                raise ValueError('expanded dialogue sound differs from original authored scenario/role/meaning')
        result['dialogues'] = deepcopy(dialogues)
        proof['original_authored_dialogue_manifest'] = {'path': path, 'sha256': expected}
        proof['all_sound_transcripts_roles_meanings_match_original_authority'] = True
    result['production_registry_header_view'] = proof
    return result
