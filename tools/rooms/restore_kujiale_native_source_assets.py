#!/usr/bin/env python3
"""Restore only task-local UE MIC backups and archive withdrawn derived assets."""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,shutil
from pathlib import Path
import psutil


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();u=root/'ue_v1';v=root/'ue_round7';content=u/'runtime/SpearSim/Content';backup=u/'native_source_material_backup_v3'
 if root.name!='v7_kujiale_20261003' or content.is_symlink():raise ValueError('restore requires the task-owned isolated project')
 for proc in psutil.process_iter(['name','cmdline']):
  if proc.info['name']=='UnrealEditor' and any(str(u) in token for token in proc.info['cmdline'] or []):raise RuntimeError('task-owned UE editor is still active; restore deferred')
 repairs=read(u/'evidence/native_source_material_repair_v3.json')['repairs']
 if len(repairs)!=75:raise ValueError('expected exact 75 native MIC backups')
 receipt={'schema':'kujiale_native_original_material_restore_v1','status':'started','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':[],'archived_derivatives':[]}
 for item in repairs:
  package=item['native_material'].split('.')[0]
  if not package.startswith('/Game/'):raise ValueError('unexpected original material package')
  relative=Path(package.removeprefix('/Game/')+'.uasset');source=backup/relative;destination=content/relative
  if source.is_symlink() or destination.is_symlink() or content not in destination.resolve().parents:raise ValueError('restore path escaped own Content')
  expected=item['before_sha256'];current=sha(destination)
  if sha(source)!=expected or current not in (expected,item['after_sha256']):raise ValueError('material backup or modified-file identity changed: '+package)
  receipt['records'].append({'native_material':item['native_material'],'path':str(destination),'backup':str(source),'restored_expected_sha256':expected,'before_restore_sha256':current})
 # All identities verified before the first reversible write. Atomic replacement
 # prevents writing through any accidental hard link into another checkout.
 for item in receipt['records']:
  destination=Path(item['path']);temporary=destination.with_name(destination.name+'.round7_restore')
  if temporary.exists():raise FileExistsError(temporary)
  shutil.copy2(item['backup'],temporary);os.replace(temporary,destination)
  item['after_restore_sha256']=sha(destination)
  if item['after_restore_sha256']!=item['restored_expected_sha256']:raise ValueError('restored native MIC bytes disagree with backup')
 archive=v/'withdrawn_material_assets';archive.mkdir(parents=True,exist_ok=True)
 for name in ('V7SourceMaterialsV2','V7SourceMaterialsV3','V7SourceMaterialsRound6V1','V7SourceMaterialsRound6V2'):
  source=content/'AVEngine'/name
  if not source.exists():continue
  if source.is_symlink() or any(x.is_symlink() for x in source.rglob('*')):raise ValueError('derived material archive contains a symbolic link')
  destination=archive/name
  if destination.exists():raise FileExistsError(destination)
  files=[x for x in source.rglob('*') if x.is_file()]
  item={'original_content_directory':str(source),'archived_directory':str(destination),'files':[{'relative_path':str(x.relative_to(source)),'bytes':x.stat().st_size,'sha256':sha(x)} for x in files]}
  source.rename(destination);receipt['archived_derivatives'].append(item)
 receipt.update(status='pass',restored_mic_count=len(receipt['records']),linear_mr_images_unbound=32,raw_assets_modified=False,shared_content_modified=False)
 write(v/'evidence/original_ue_material_restore.json',receipt)
 with (root/'PROGRESS_zh.md').open('a') as f:f.write('\n\n## '+receipt['time_utc']+' 原生 UE 材质恢复完成\n\n- 75个本任务MIC从修改前备份逐字节恢复并核对sha256；32张派生MR不再被原MIC引用，派生材质/MR从自己的Content移到ue_round7/withdrawn_material_assets保留，不参与渲染。原始资产和共享Content未写。证据ue_round7/evidence/original_ue_material_restore.json。\n')
 print('RESTORE',receipt['status'],receipt['restored_mic_count'],receipt['linear_mr_images_unbound'])

if __name__=='__main__':main()
