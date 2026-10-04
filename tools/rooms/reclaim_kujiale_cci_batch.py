"""Reclaim only exact owned bytes after independently checked CCI SHA receipts."""
import argparse,hashlib,json,time,os
from pathlib import Path
from tools.rooms.alpha_storage_contract import write_owned_json as write_json
from tools.rooms.alpha_storage_contract import native_seal,relative_file
from tools.rooms.prepare_kujiale_cci_batch import digest


def reclaim_batch(plan_path,receipt_path,task_root):
    root=Path(task_root).resolve()/'ue_round8';plan_path=Path(plan_path).resolve();receipt_path=Path(receipt_path).resolve()
    if plan_path.parent!=root/'evidence' or receipt_path.parent!=root/'evidence':raise ValueError('receipts must be task-owned')
    plan=json.loads(plan_path.read_text());receipt=json.loads(receipt_path.read_text())
    if receipt['status']!='all_SHA256_verified' or receipt['cci_base']!='/mnt/afs/xinyuanqian/jzy/hardbank/mirror':raise ValueError('missing verified CCI mirror receipt')
    by_path={r['path']:r for r in receipt['files']};expected={receipt['cci_base']+'/'+r['cci_relative']:r for r in plan['source_files']}
    if set(by_path)!=set(expected):raise ValueError('CCI receipt file set differs from this exact batch')
    delete={};seal_rows=[];bundle_rows=plan.get('bundles',[{'directory':plan.get('bundle')}])
    for path,row in expected.items():
        proof=by_path[path]
        if proof['sha256']!=row['sha256'] or proof['bytes']!=row['bytes']:raise ValueError('CCI checksum or size differs')
        source=Path(row['source']);source.relative_to(root/'deliveries')
        p=relative_file(root/'deliveries',source.relative_to(root/'deliveries'))
        if digest(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise ValueError('48g source changed since batch freeze')
        delete[str(p)]=row
    for b in bundle_rows:
        bundle=Path(b['directory']).resolve()
        if bundle.parent!=root/'deliveries' or bundle.is_symlink():raise ValueError('unexpected bundle directory')
        if {str(p) for p in bundle.rglob('*') if p.is_file()}!={str(p) for p in delete if Path(p).is_relative_to(bundle)}:raise ValueError('unlisted bundle data would be removed')
        portable=bundle/'PORTABILITY.json'
        if portable.exists():
            meta=json.loads(portable.read_text());seal=Path(meta['source_seal']).resolve();parent,recipe,pointer,source=native_seal(seal.parent,task_root)
            if seal!=source:raise ValueError('source seal differs from the admitted parent')
            compact=json.loads((parent/'alpha_compaction.json').read_text())
            if compact['status']!='compacted':raise ValueError('parent intermediates not compacted')
            inventory=compact['preserved_delivery_files']
            if {str(p) for p in source.rglob('*') if p.is_file()}!={str(parent/r['path']) for r in inventory}:raise ValueError('source seal inventory changed')
            for row in inventory:
                p=relative_file(parent,row['path'])
                if digest(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise ValueError('original sealed bytes changed')
                delete[str(p)]={'source':str(p),'bytes':row['bytes'],'sha256':row['sha256']}
            seal_rows.append({'parent':str(parent),'source_seal':str(source)})
    record=plan_path.with_name(plan_path.stem+'_reclaim.json')
    if record.exists():raise FileExistsError('reclaim receipt already exists; inspect before any resume')
    before=os.statvfs(root);allocated={}
    for p in delete:
        st=Path(p).stat();allocated[(st.st_dev,st.st_ino)]=st.st_blocks*512
    value={'status':'all_verified_ready_to_delete','plan_sha256':digest(plan_path),'cci_receipt_sha256':digest(receipt_path),'verified_CCI_files':len(expected),'logical_bytes':sum(r['bytes'] for r in delete.values()),'unique_inode_allocated_bytes_upper_bound':sum(allocated.values()),'deletion_inventory':list(delete.values()),'source_seals':seal_rows,'sample_epoch':time.time()}
    prepared=record.with_name(record.stem+'_prepared.json')
    if prepared.exists():raise FileExistsError('prepared reclaim exists; inspect interrupted action before resume')
    write_json(prepared,value)
    # No broad rmtree: every file has a checked SHA; only emptied listed trees
    # are removed. Shared/raw source data, old rounds and witnesses are retained.
    for p in delete:Path(p).unlink()
    for directory in [Path(b['directory']) for b in bundle_rows]+[Path(s['source_seal']) for s in seal_rows]:
        for d in sorted([p for p in directory.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):d.rmdir()
        directory.rmdir()
    for row in seal_rows:
        parent=Path(row['parent']);write_json(parent/'CCI_TRANSFER_VERIFIED.json',{'reclaim_receipt':str(record),'cci_receipt':str(receipt_path),'source_seal':row['source_seal'],'status':'transferred_SHA256_verified_local_delivery_reclaimed'})
    after=os.statvfs(root);value.update(status='reclaimed',shared_free_before_bytes=before.f_bavail*before.f_frsize,shared_free_after_bytes=after.f_bavail*after.f_frsize)
    write_json(record,value);return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['plan','cci-receipt','task-root']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();value=reclaim_batch(a.plan,a.cci_receipt,a.task_root);print(json.dumps({k:value[k] for k in ['status','verified_CCI_files','logical_bytes','unique_inode_allocated_bytes_upper_bound']}))
if __name__=='__main__':main()
