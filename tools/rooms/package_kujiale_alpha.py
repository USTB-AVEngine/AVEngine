#!/usr/bin/env python3
"""Publish portable alpha delivery without changing questions or any answers."""
from __future__ import annotations
import argparse,ctypes,errno,hashlib,json,os,shutil,time,uuid
from pathlib import Path
from tools.rooms.alpha_storage_contract import write_owned_json as write_json
from tools.rooms.alpha_storage_contract import native_seal


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def publish_noreplace(temp,target):
    libc=ctypes.CDLL(None,use_errno=True)
    if not hasattr(libc,'renameat2'):raise RuntimeError('atomic no-replace directory publication requires Linux renameat2')
    call=libc.renameat2;call.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    if call(-100,os.fsencode(temp),-100,os.fsencode(target),1):
        error=ctypes.get_errno();raise OSError(error,os.strerror(error),str(target))


def package_alpha(result,parent,task_root):
    task_root=Path(task_root).resolve();scope=task_root/'ue_round8'
    parent,_recipe,seal,source=native_seal(parent,task_root)
    if result['status']!='formal_alpha_admitted':raise ValueError('completed native alpha parent required')
    out=scope/'deliveries';out.mkdir(exist_ok=True);target=out/(parent.name+"__portable_v2")
    if target.exists():raise FileExistsError('portable delivery already exists; no overwrite permitted')
    temp=out/('.partial_'+parent.name+'_v2_'+uuid.uuid4().hex);temp.mkdir();rewrites=[]
    for f in source.rglob('*'):
        if not f.is_file():continue
        if any(part.startswith('.') for part in f.relative_to(source).parts):continue
        if f.is_symlink():raise ValueError('seal contains a symlink')
        dst=temp/f.relative_to(source);dst.parent.mkdir(parents=True,exist_ok=True)
        os.link(f,dst)
    # Private readback links point to retained bundle evidence, never removed
    # stage attempts on 48g. The original evidence remains immutable upstream.
    for row in result['windows']:
        wid=row['window_id'];v,factor=row['member'].split('_');p=temp/'private'/(wid+'_evidence.json')
        evidence=json.loads(p.read_text());changes={
            'native_visual_evidence':'native/'+v+'/native_alpha_visual_evidence.json',
            'native_audio_evidence':'native/'+factor+'_audio_evidence.json'}
        for key,value in changes.items():
            if key in evidence:rewrites.append({'window':wid,'field':key,'old_48g_path':evidence[key],'portable_path':value});evidence[key]=value
        proof=evidence.get('source_qa_evidence',{}).get('foa_readback')
        if proof:
            value='../windows/'+wid+'/audio_foa.wav';rewrites.append({'window':wid,'field':'source_qa_evidence.foa_readback.path','old_48g_path':proof['path'],'portable_path':value});proof['path']=value
        p.unlink();write_json(p,evidence)
    licenses=json.loads((temp/'private/human_license_admission.json').read_text())
    for row in licenses:
        license=row['license'];src=Path(license['retained_text']);dst=temp/'private/licenses'/(license['sha256']+'.md');dst.parent.mkdir(exist_ok=True)
        if not dst.exists():shutil.copyfile(src,dst)
        if sha(dst)!=license['sha256']:raise ValueError('retained MIT license differs')
    for row in licenses:
        row['license']['retained_text']='licenses/'+row['license']['sha256']+'.md'
    license_path=temp/'private/human_license_admission.json'
    license_path.unlink();write_json(license_path,licenses)
    checks=[]
    for row in result['windows']:
        wid=row['window_id']
        for rel in ['windows/'+wid+'/questions.json','private/'+wid+'_answers.json']:
            checks.append({'path':rel,'source_sha256':sha(source/rel),'delivery_sha256':sha(temp/rel)})
    if any(r['source_sha256']!=r['delivery_sha256'] for r in checks):raise ValueError('questions or answers changed while packaging')
    proof={'schema':'kujiale_portable_alpha_bundle_v2','source_seal':str(source),'source_profile':result['profile'],'scene_id':parent.name,
           'alpha_formal_windows':result['alpha_formal_windows'],'full_v7_formal_windows':0,'question_answer_byte_identity':checks,
           'artifact_path_rewrites':rewrites,'raw_InteriorAgent_or_native_UE_assets_included':False,'release_use':'benchmark_only' if '_bench_' in parent.name else 'formal_alpha_canary','sample_epoch':time.time()}
    write_json(temp/'PORTABILITY.json',proof)
    manifest=[{'path':str(f.relative_to(temp)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(temp.rglob('*')) if f.is_file()]
    write_json(temp/'BUNDLE_MANIFEST.json',{'schema':'kujiale_portable_alpha_manifest_v2','files':manifest,'logical_bytes':sum(r['bytes'] for r in manifest)})
    # Flush payload files before the atomic directory publication.
    for f in temp.rglob('*'):
        if f.is_file():
            with f.open('rb') as stream:os.fsync(stream.fileno())
    publish_noreplace(temp,target)
    receipt={'status':'packaged','directory':str(target),'manifest_sha256':sha(target/'BUNDLE_MANIFEST.json'),
             'logical_bytes':sum(f.stat().st_size for f in target.rglob('*') if f.is_file()),'question_answer_unchanged':True,'parent':str(parent)}
    write_json(parent/'portable_delivery_v2.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['result','parent','task-root']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(package_alpha(json.loads(a.result.read_text()),a.parent,a.task_root),ensure_ascii=False))
if __name__=='__main__':main()
