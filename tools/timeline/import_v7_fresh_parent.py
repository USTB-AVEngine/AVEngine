#!/usr/bin/env python3
"""Revalidate one explicitly authorized fresh helper parent into acceptance."""
from pathlib import Path
import argparse,datetime,fcntl,json,os,time
from tools.timeline.export_v7_trial import parent_payload,load,write
from avengine.timeline.v7_s_batch_checks import open_delivered_media
from avengine.timeline.v7_generation_scope import assert_new_generation_output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--source-receipt',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);a=p.parse_args()
    m=load(a.manifest);root=Path(m['output_root']).resolve();source=a.source_root.resolve();assert_new_generation_output(root)
    if str(source)not in m.get('authorized_fresh_source_roots',[]):raise ValueError('fresh helper source root not explicitly authorized')
    if not a.source_receipt.resolve().is_relative_to(source):raise ValueError('source receipt escaped read-only helper root')
    native=load(a.source_receipt)
    if native.get('status')!='passed'or native.get('reader',{}).get('status')!='pass':raise ValueError('helper native producer and actual media reader must both pass')
    pointer_path=Path(native['export']['source_seal']);pointer=load(pointer_path);recipe=pointer['recipe']['input_recipe'];row=recipe.get('row',recipe.get('seed_row'));scene=row['scene_id']
    if not pointer_path.resolve().is_relative_to(source):raise ValueError('source seal escaped the authorized fresh root')
    started=time.monotonic()
    with(root/'inputs/plan.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);m=load(a.manifest)
        matches=[t for t in m['parents']if t['row']['scene_id']==scene]
        if matches:
            slot=matches[0]['slot'];path=root/'evidence/jobs'/f'{slot:03d}.json'
            if path.exists()and load(path).get('status')=='passed':return
        else:
            slot=len(m['parents']);family=native.get('family')or('kujiale'if row['house'].startswith('kujiale')else'mp3d'if row['house'].startswith('mp3d')else'hm3d')
            m['parents'].append({'slot':slot,'family':family,'device':'ue'if family=='kujiale'else'cpu','profile':'registered'if len(row['members'])==4 else'phase2',
                'focus':'content_binding'if len(row['members'])==4 else'motion_quartet_template','row':row,'launch_disabled':True,'read_only_helper_receipt':str(a.source_receipt)})
            write(a.manifest,m)
    info={'status':'passed','house':row['house'],'seal_path':str(pointer_path)}
    rows,export=parent_payload(info,root/'export',root,read_only_source_roots=[source])
    for r in rows:r['dataset_role']='acceptance_only'
    table=root/'export/parent_rows'/f'{slot:03d}.jsonl';table.parent.mkdir(parents=True,exist_ok=True);table.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in rows))
    # Existing reader sees the exact new exported payload, all windows/streams.
    readroot=root/'reader_checks_import'/f'{slot:03d}';readroot.mkdir(parents=True,exist_ok=True)
    media=[]
    for win in export['windows']:
        folder=Path('parents')/scene/'windows'/win['window_id']
        for name in ('video.mp4','audio_foa.wav','audio_mono.wav'):
            target=readroot/folder/name;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():os.link(root/'export'/folder/name,target)
        media.append({'window_id':win['window_id'],'video':str(folder/'video.mp4'),'audio_foa':str(folder/'audio_foa.wav'),'audio_mono':str(folder/'audio_mono.wav')})
    (readroot/'train.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in media));reader=open_delivered_media(readroot)
    if reader['status']!='pass':raise ValueError('fresh imported media reader failed')
    receipt={**native,'slot':slot,'status':'passed','scene_id':scene,'house':row['house'],'room_id':row['room_id'],'export':export,'reader':reader,
        'origin_helper_receipt':str(a.source_receipt),'origin_helper':source.name,'original_native_code_commit':native.get('code_commit'),
        'independent_questions':export['independent_questions'],'raw_questions':export['raw_export_questions'],
        'fresh_import_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fresh_import_seconds':time.monotonic()-started,'old_media_or_seals_reused':False}
    write(root/'evidence/jobs'/f'{slot:03d}.json',receipt)
    print(json.dumps({'slot':slot,'scene_id':scene,'independent_questions':receipt['independent_questions'],'policy':export['scene_delivery_policy']['status']}),flush=True)

if __name__=='__main__':main()
