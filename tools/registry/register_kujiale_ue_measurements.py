#!/usr/bin/env python3
"""Promote complete current UE readbacks into the production registration."""
from __future__ import annotations
import argparse,datetime,hashlib,json
from pathlib import Path
import numpy as np
from avengine.registry.kujiale_ue_measurements import SCHEMA,sha256,loaded_map_package

def load(path):return json.loads(Path(path).read_text())
def build(manifest):
    root=Path(manifest['output_root']);houses=load(manifest['houses_ue'])['houses'];rooms={}
    for h in houses:
        for rid in h['checked_room_ids']:rooms[h['house']+'/'+rid]={'house':h['house'],'room_id':rid,'measurement_status':'not_run','smoke_status':'not_run','measurements':[],'independent_questions':0,'reason':'current full UE capture and parent smoke pending'}
    assets={};jobs={}
    for path in sorted((root/'evidence/jobs').glob('[0-9][0-9][0-9].json')):
        job=load(path);task=next((t for t in manifest['parents']if t['slot']==job['slot']),None)
        if task is None:continue
        row=task['row'];key=row['house']+'/'+row['room_id'];entry=rooms[key]
        if job['status']=='passed'and entry['smoke_status']!='pass':
            entry.update(smoke_status='pass',independent_questions=job['independent_questions'],scene_id=job['scene_id'],families=job['export']['question_family_counts'],parent_receipt={'path':str(path),'sha256':sha256(path)},reader=job['reader'],english_status='pass',reason='')
        elif entry['smoke_status']!='pass':entry.update(smoke_status=job['status'],reason=job.get('reason','current complete parent pending'),scene_id=job['scene_id'])
        native=Path(task.get('native_output_root')or root/'native')/'parents'/row['scene_id']
        house=next(h for h in houses if h['house']==row['house'])
        for marker in sorted(native.glob('visual_*_completed.json')):
            pointer=load(marker);path=native/pointer['attempt']/'native_registered_visual_evidence.json'
            if not path.exists():continue
            visual=load(path);frames=visual['frames']
            if visual['frame_count']!=450 or len(frames)!=450:raise ValueError('incomplete current native frames '+str(path))
            local=load(visual['measured_registration_copy']);by={a['asset_id']:a for a in local['assets']};first=[];max_error=0.
            for frame in frames:
                for actor in frame['actors']:
                    aid=actor['asset_id']
                    if aid not in by:continue
                    asset=by[aid];matrix=np.asarray(actor['body_to_world_m'],float)
                    expected=(matrix@np.r_[asset['mouth_local_glb_m'],1.])[:3]
                    error=float(np.linalg.norm(expected-actor['emitter_position_m']));max_error=max(max_error,error)
                    if error>.002:raise ValueError('unchanged 2mm native mouth gate failed')
            for actor in frames[0]['actors']:
                aid=actor['asset_id']
                if aid not in by:continue
                proof=load(actor['native_load_receipt'])
                if proof['status']!='pass':raise ValueError('native package load did not pass')
                packages=[]
                for package in proof['files']:
                    if sha256(package['path'])!=package['sha256']or sha256(package['source'])!=package['sha256']:raise ValueError('registered native package bytes changed')
                    packages.append({'object_path':package['object_path'],'source':str(Path(package['source']).resolve()),'sha256':package['sha256']})
                new={k:proof[k]for k in ['asset_id','actual_class_path','actual_mesh_paths','loaded_artifact_sha256','original_glb_sha256','glb_to_ue_local_cm','mouth_local_glb_m','mouth_local_ue_cm']};new['packages']=packages
                if aid in assets and assets[aid]!=new:raise ValueError('inconsistent measured native identity '+aid)
                assets[aid]=new;first.append({'asset_id':aid,'first_body_to_world_m':actor['body_to_world_m'],'loaded_artifact_sha256':actor['loaded_artifact_sha256'],'load_receipt':{'path':actor['native_load_receipt'],'sha256':sha256(actor['native_load_receipt'])}})
            launch=visual['renderer']['launch'];map_receipt=loaded_map_package(house['input']['ue_config']['uproject'],house['input']['ue_map_path'],launch['temporary_directory'])
            if not any(r['visual_evidence']['path']==str(path)for r in entry['measurements']):entry['measurements'].append({'scene_id':row['scene_id'],'visual':marker.stem,'frames_checked':450,'mouth_max_error_m':max_error,'actors':first,'map_package':map_receipt,'actual_graphics_gpu':launch['actual_graphics_gpu'],'visual_evidence':{'path':str(path),'sha256':sha256(path)}})
            entry['measurement_status']='pass'
    return {'schema':SCHEMA,'revision':'20261015-native-measured-v1','updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_registrations_modified':False,'runtime_authority':'production reads these measured native bases and original package hashes; world matrices remain measured per new frame','scope':{'houses':len(houses),'rooms':len(rooms)},'assets':list(assets.values()),'rooms':list(rooms.values())}

def publish(manifest,path):
    data=build(manifest);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temporary.replace(path);path.with_suffix('.json.sha256').write_text(sha256(path)+'  '+path.name+'\n')
    root=Path(manifest['output_root']);passed=[r for r in data['rooms']if r['smoke_status']=='pass'];text=['全部33间房尚未完成。'if len(passed)<len(data['rooms'])else'全部33间房已实测并完整通过。','','更新：'+data['updated_utc']+'；正式登记：`'+str(path)+'`，sha256：`'+sha256(path)+'`。','',f"完整通过 {len(passed)}/{len(data['rooms'])} 间；实测 {sum(r['measurement_status']=='pass'for r in data['rooms'])} 间；去重独立题 {sum(r['independent_questions']for r in passed)} 道。数字仅计每间房一个通过父场景，带狗同房专项不重复充房数。",'','|房子/房间|UE测量|完整冒烟|去重题数|原因/修法|','|---|---|---|---:|---|']
    for r in data['rooms']:text.append('|'+r['house']+'/'+r['room_id']+'|'+r['measurement_status']+'|'+r['smoke_status']+'|'+str(r['independent_questions'])+'|'+r['reason'].replace('|','/')+'|')
    text+=['','完整通过表示原生产14题及5绑定门禁、四成员五半重叠窗去重、英文检查和真实媒体reader均成功；未跑/运行中/失败不算能用。逐帧矩阵与原始包身份、实测加载地图和UUID证据在正式登记各rooms.measurements中。','GPU1/3枚举正确不等于UE画面来自对应卡；以实际UE日志、精确PID/NVML和capture读回成功为准。','既有失败和检查点均保留；未删除原始数据，未push/CCI/ACP。']
    (root/'ALL_ROOMS_zh.md').write_text('\n'.join(text)+'\n');return data

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--registration',type=Path,required=True);args=parser.parse_args();data=publish(load(args.manifest),args.registration);print(json.dumps({'houses':data['scope']['houses'],'rooms':data['scope']['rooms'],'measured':sum(r['measurement_status']=='pass'for r in data['rooms']),'passed':sum(r['smoke_status']=='pass'for r in data['rooms']),'sha256':sha256(args.registration)}))
if __name__=='__main__':main()
