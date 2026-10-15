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
    contract=load(manifest['contract']);legality_source=contract['inputs']['room_kujiale'];legality=load(legality_source['path']);legal_rows=legality['rooms'].values()if isinstance(legality['rooms'],dict)else legality['rooms'];floor_by_room={(r['house'],r['room_id']):r['native_floor_surface']for r in legal_rows}
    for h in houses:
        for rid in h['checked_room_ids']:rooms[h['house']+'/'+rid]={'house':h['house'],'room_id':rid,'measurement_status':'not_run','smoke_status':'not_run','measurements':[],'independent_questions':0,'reason':'current full UE capture and parent smoke pending'}
    for entry in rooms.values():entry['native_floor_surface']=floor_by_room[(entry['house'],entry['room_id'])];entry['room_legality_source']={k:legality_source[k]for k in ['path','sha256']}
    from avengine.registry.kujiale_ue_measurements import DEFAULT_REGISTRATION
    bootstrap=Path(manifest.get('formal_kujiale_registration',DEFAULT_REGISTRATION));assets={a['asset_id']:a for a in load(bootstrap).get('assets',[])}if bootstrap.exists()else{};jobs={}
    def repair(reason):
        if 'support face is not room floor'in reason:return '已接通正式登记内原有逐三角形地板来源；保持原身份/接触检查，自动重跑'
        if 'geometry construction'in reason:return '同房细化网格、真实地板基准和最多128个原生Nav镜头自动搜索'
        if 'asset category'in reason:return '先完整UE测量新类别见证，再生成新契约并复走原生产门禁'
        if 'binding questions'in reason:return '自动构造异外观双说话人、静默旁观者与完整错开发声TTS四成员，保留5门槛'
        if 'contact'in reason:return '原生地板/深度驱动固定根偏移；不可满足则同房下一候选，标准不变'
        if 'vision:'in reason:return '自动补实体结构引用与实测距离/入画候选；原视觉配额不变'
        return '按原生日志定位并自动重构同房下一条件；未通过不计能用'
    for path in sorted((root/'evidence/jobs').glob('[0-9][0-9][0-9].json')):
        job=load(path);task=next((t for t in manifest['parents']if t['slot']==job['slot']),None)
        if task is None:continue
        row=task['row'];key=row['house']+'/'+row['room_id'];entry=rooms[key]
        delivery=None
        if job['status']=='passed':
            from avengine.timeline.v7_export_scene_policy import cached_parent_policy
            delivery=cached_parent_policy(job['export']['source_seal'],root/'evidence/current_owner_policy_cache')
        valid_smoke=job['status']=='passed'and delivery['status']=='pass'
        if valid_smoke and entry['smoke_status']!='pass':
            summary={k:v for k,v in delivery.items()if k in {'revision','status','source_seal','reasons','member_listener','original_native_evidence'}};summary['native_layout']={v:proof['layout']for v,proof in delivery['native_views'].items()if v!='windows'}
            entry.update(smoke_status='pass',independent_questions=job['independent_questions'],scene_id=job['scene_id'],families=job['export']['question_family_counts'],parent_receipt={'path':str(path),'sha256':sha256(path)},reader=job['reader'],english_status='pass',current_owner_scene_policy=summary,reason='')
        elif entry['smoke_status']!='pass':
            entry.update(smoke_status='failed'if delivery and not valid_smoke else job['status'],reason=';'.join(delivery['reasons'])if delivery and not valid_smoke else job.get('reason','current complete parent pending'),scene_id=job['scene_id'])
        if valid_smoke and job.get('new_current_native_category_qualification'):
            qualification=job['new_current_native_category_qualification'];all_qualifications=entry.setdefault('current_native_category_qualifications',[])
            if qualification not in all_qualifications:all_qualifications.append(qualification)
            entry.setdefault('current_native_category_qualification',qualification)
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
    for entry in rooms.values():entry['fix']='完整门禁已过'if entry['smoke_status']=='pass'else repair(entry['reason'])
    return {'schema':SCHEMA,'revision':'20261015-native-measured-v1','updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_registrations_modified':False,'runtime_authority':'production reads these measured native bases and original package hashes; world matrices remain measured per new frame','scope':{'houses':len(houses),'rooms':len(rooms)},'assets':list(assets.values()),'rooms':list(rooms.values())}

def publish(manifest,path):
    data=build(manifest);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temporary.replace(path);path.with_suffix('.json.sha256').write_text(sha256(path)+'  '+path.name+'\n')
    root=Path(manifest['output_root']);passed=[r for r in data['rooms']if r['smoke_status']=='pass'];text=['全部33间房尚未完成。'if len(passed)<len(data['rooms'])else'全部33间房已实测并完整通过。','','更新：'+data['updated_utc']+'；正式登记：`'+str(path)+'`，sha256：`'+sha256(path)+'`。','',f"完整通过 {len(passed)}/{len(data['rooms'])} 间；实测 {sum(r['measurement_status']=='pass'for r in data['rooms'])} 间；去重独立题 {sum(r['independent_questions']for r in passed)} 道。数字仅计每间房一个通过父场景，带狗同房专项不重复充房数。",'','|房子/房间|UE测量|完整冒烟|去重题数|原因/修法|','|---|---|---|---:|---|']
    for r in data['rooms']:text.append('|'+r['house']+'/'+r['room_id']+'|'+r['measurement_status']+'|'+r['smoke_status']+'|'+str(r['independent_questions'])+'|'+(r['reason']+'；'+r['fix']).replace('|','/')+'|')
    text+=['','完整通过表示原生产14题及5绑定门禁、四成员五半重叠窗去重、英文检查和真实媒体reader均成功；未跑/运行中/失败不算能用。逐帧矩阵与原始包身份、实测加载地图和UUID证据在正式登记各rooms.measurements中。','GPU1/3枚举正确不等于UE画面来自对应卡；以实际UE日志、精确PID/NVML和capture读回成功为准。','既有失败和检查点均保留；未删除原始数据，未push/CCI/ACP。']
    (root/'ALL_ROOMS_zh.md').write_text('\n'.join(text)+'\n');return data

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--registration',type=Path,required=True);args=parser.parse_args();data=publish(load(args.manifest),args.registration);print(json.dumps({'houses':data['scope']['houses'],'rooms':data['scope']['rooms'],'measured':sum(r['measurement_status']=='pass'for r in data['rooms']),'passed':sum(r['smoke_status']=='pass'for r in data['rooms']),'sha256':sha256(args.registration)}))
if __name__=='__main__':main()
