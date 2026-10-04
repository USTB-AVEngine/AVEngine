"""Freeze exact owned derived-file list before invoking the provided CCI pull."""
import argparse,hashlib,json,time
from pathlib import Path
from tools.rooms.alpha_storage_contract import write_owned_json as write_json
from tools.rooms.alpha_storage_contract import relative_file


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def prepare_batch(task_root,bundles,name):
    root=Path(task_root).resolve()/'ue_round8';base=root/'deliveries'
    if not name.replace('_','').isalnum():raise ValueError('unsafe batch name')
    rows=[];bundle_rows=[]
    for supplied in bundles:
        bundle=Path(supplied)
        if bundle.is_symlink():raise ValueError('bundle symlink')
        bundle=bundle.resolve()
        if bundle.parent!=base:raise ValueError('bundle outside owned derived delivery')
        manifest=relative_file(bundle,'BUNDLE_MANIFEST.json');m=json.loads(manifest.read_text())
        for f in m['files']:
            p=relative_file(bundle,f['path'])
            if digest(p)!=f['sha256'] or p.stat().st_size!=f['bytes']:raise ValueError('source bundle changed')
        files=sorted(p for p in bundle.rglob('*') if p.is_file())
        if set(str(p.relative_to(bundle)) for p in files)!={f['path'] for f in m['files']}|{'BUNDLE_MANIFEST.json'}:raise ValueError('undeclared bundle file')
        for p in files:
            if p.is_symlink():raise ValueError('bundle contains a symlink')
            rows.append({'source':str(p),'cci_relative':str(p).lstrip('/'),'bytes':p.stat().st_size,'sha256':digest(p)})
        bundle_rows.append({'directory':str(bundle),'manifest_sha256':digest(manifest)})
    if len({r['source'] for r in rows})!=len(rows):raise ValueError('duplicate batch file')
    path=root/'evidence'/('transfer_'+name+'.json');list_path=path.with_suffix('.list')
    if path.exists() or list_path.exists():raise FileExistsError('batch publication cannot overwrite')
    value={'schema':'kujiale_cci_transfer_plan_v1','bundles':bundle_rows,'source_files':rows,'logical_bytes':sum(r['bytes'] for r in rows),'sample_epoch':time.time(),'raw_assets_included':False}
    write_json(path,value);list_path.write_text(''.join(r['cci_relative']+'\n' for r in rows))
    return {'plan':str(path),'list':str(list_path),'files':len(rows),'logical_bytes':value['logical_bytes']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task-root',type=Path,required=True);p.add_argument('--bundles',type=Path,nargs='+',required=True);p.add_argument('--name',required=True);a=p.parse_args();print(json.dumps(prepare_batch(a.task_root,a.bundles,a.name)))
if __name__=='__main__':main()
