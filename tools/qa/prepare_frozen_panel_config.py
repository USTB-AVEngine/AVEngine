#!/usr/bin/env python3
"""Build a pilot/test configuration from explicit local runtime and admitted rooms.

Does not copy/modify datasets or query GPUs. CCI paths, staged speech/assets and
accepted-house packages must already exist; no implicit 48g path substitution.
"""
import argparse
import json
from pathlib import Path

def read(path):return json.loads(Path(path).read_text())
def require(value,reason):
    if not value:raise ValueError(reason)
def build(base,houses,scope,*,panel_id,split,per_house,expected_houses):
    require(len(houses)==expected_houses and len({h['house'] for h in houses})==len(houses),'accepted house count/identity differs')
    allowed={h['house']:h for h in scope}
    for house in houses:
        require(house['house'] in allowed,'house missing from reviewed scope')
        rooms={r['room_label']:r for r in allowed[house['house']]['checked_rooms']}
        require(bool(house['rooms']),'house has no admitted rooms')
        require(all(r['room_label'] in rooms and r==rooms[r['room_label']] for r in house['rooms']),'room placement differs from reviewed scope')
        accepted=read(house['acceptance_result'])
        require(accepted['status']=='pass','house acceptance not pass')
        for key in ('navmesh_source','semantic_source','annotation_source'):require(Path(accepted['input'][key]).is_file(),f'local accepted source unavailable: {key}')
        require((Path(accepted['input']['scene_directory'])/(accepted['input']['scan_id']+'.glb')).is_file(),'local accepted scan unavailable')
        for room in house['rooms']:
            require(any(r['room_label']==room['room_label'] and r.get('floor_polygon') for r in accepted['input']['acoustic_rooms']),'accepted room has no measured floor polygon')
        for key in ('room_manifest','acoustic_package_manifest'):require(Path(house[key]).is_file(),f'house {key} unavailable')
    require(split in {'pilot','test'} and per_house>0,'invalid panel protocol')
    require(split!='test' or expected_houses==12 and per_house==60,'formal panel requires 12 houses x 60 groups')
    for key in ('template_plan','template_dataset','speech','registry','hrtf'):require(Path(base[key]).is_file(),f'local input unavailable: {key}')
    for key in ('runtime_prefix','magnum_python_site','rlr_sdk_root','mp3d_root'):require(Path(base['runtime'][key]).exists(),f'local runtime unavailable: {key}')
    speech=read(base['speech']);require(all(Path(s['path']).is_file() for s in speech['sounds']),'speech contains unmigrated dry paths')
    return {**base,'houses':houses,'panel_id':panel_id,'split':split,'groups_per_house':per_house,
            'group_count':len(houses)*per_house,'min_free_vram_mb':max(12288,base.get('min_free_vram_mb',0)),
            'primary_qa':'QA-27','model_inference':'not_run'}
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--base-config',type=Path,required=True);p.add_argument('--houses',type=Path,required=True)
    p.add_argument('--scope',type=Path,required=True);p.add_argument('--panel-id',required=True);p.add_argument('--split',choices=['pilot','test'],required=True)
    p.add_argument('--groups-per-house',type=int,required=True);p.add_argument('--expected-houses',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--pilot-verification',type=Path)
    a=p.parse_args();houses=read(a.houses);houses=houses['houses'] if isinstance(houses,dict) else houses
    c=build(read(a.base_config),houses,read(a.scope),panel_id=a.panel_id,split=a.split,per_house=a.groups_per_house,expected_houses=a.expected_houses)
    if a.split=='test':
        require(a.pilot_verification is not None,'formal panel requires the verified pilot exclusion list')
        require(a.seed!=c['seed'],'formal panel requires a new seed')
        previous=read(a.pilot_verification);require(previous['status']=='pass' and len(previous['groups'])==20,'pilot exclusion requires 20 verified groups')
        c['excluded_physical_parent_sha256']=[r['physical_parent_sha256'] for r in previous['groups']]
        c['excluded_video_pair_sha256']=[r['video_pair_sha256'] for r in previous['groups']]
    c['seed']=a.seed
    with a.output.open('x',encoding='utf8') as f:json.dump(c,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':'pass','house_count':len(houses),'independent_groups':c['group_count'],'config':str(a.output)},ensure_ascii=False))
if __name__=='__main__':main()
