"""Read-only file allocation accounting; deletion proposals never unlink files."""
from __future__ import annotations
import argparse,collections,csv,gzip,json,os,stat,time,zlib
from pathlib import Path

ORDER=['formal_media','formal_evidence','rir','material_cache','dry_arrays',
       'full_scene_media','window_work_media','candidate_pools','native_work_evidence',
       'code_and_inputs','other']
REBUILDABLE={'material_cache','dry_arrays','full_scene_media','window_work_media','candidate_pools'}

def category(rel):
    p=Path(rel);parts=p.parts;n=p.name
    formal=any(x.startswith('alpha_seal_attempt_')or x.startswith('release')or x=='deliveries'for x in parts)
    if 'rir_cache'in parts:return 'rir'
    if 'source_material_cache'in parts:return 'material_cache'
    if formal and p.suffix.lower()in {'.wav','.mp4','.flac'}:return 'formal_media'
    if formal:return 'formal_evidence'
    if any(x.startswith('dry_attempt_')for x in parts)and p.suffix=='.npy':return 'dry_arrays'
    if n in {'scene_video.mp4','scene_foa_full_tail.wav'}:return 'full_scene_media'
    if p.suffix.lower()in {'.mp4','.wav'}and any(x.startswith(('windows_','selected_windows_'))for x in parts):return 'window_work_media'
    if n=='candidate_audit.json':return 'candidate_pools'
    if any(x.startswith(('visual_','audio_','dry_','windows_','selected_windows_'))for x in parts)or n.endswith('_completed.json')or n in {'input_recipe.json','counterfactual_alpha_audit.json'}:return 'native_work_evidence'
    if any(x in {'code','inputs','sources','licenses'}for x in parts):return 'code_and_inputs'
    return 'other'

def inspect(roots,output):
    """Metadata only. stat blocks are allocation estimates, not reflink extent ownership."""
    start=time.time();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rootpaths={k:Path(v).resolve()for k,v in roots.items()}
    if any(out==v for v in rootpaths.values()):raise ValueError('accounting output cannot replace an input root')
    by_inode={};dirs={};errors=[];symlinks=[];visited=0;last=0
    for label,root in rootpaths.items():
        if not root.is_dir():errors.append({'root':str(root),'reason':'missing root'});continue
        for base,sub,names in os.walk(root,followlinks=False):
            base=Path(base)
            sub[:]=[n for n in sub if not(base/n).is_symlink()and (base/n).resolve()!=out]
            try:s=base.stat();dirs[(s.st_dev,s.st_ino)]=s.st_blocks*512
            except OSError as e:errors.append({'path':str(base),'reason':str(e)});continue
            for n in names:
                p=base/n
                try:s=p.lstat()
                except OSError as e:errors.append({'path':str(p),'reason':str(e)});continue
                if stat.S_ISLNK(s.st_mode):symlinks.append(str(p));continue
                if not stat.S_ISREG(s.st_mode):continue
                rel=str(p.relative_to(root));c=category(rel)
                record={'root':label,'path':str(p),'relative':rel,'category':c}
                key=(s.st_dev,s.st_ino)
                value=by_inode.setdefault(key,{'allocated':s.st_blocks*512,'logical':s.st_size,'nlink':s.st_nlink,'paths':[]})
                if value['logical']!=s.st_size:errors.append({'path':str(p),'reason':'file changed size during metadata scan'})
                value['paths'].append(record);visited+=1
            if time.time()-last>=30:
                state={'status':'scanning','root':label,'file_paths_seen':visited,'elapsed_seconds':time.time()-start}
                (out/'SCANNING.json').write_text(json.dumps(state));print(json.dumps(state),flush=True);last=time.time()
    root_totals={k:collections.Counter()for k in roots};global_totals=collections.Counter();parents={};proposal=[]
    inventory=out/'FILE_ALLOCATION.jsonl.gz'
    with gzip.open(inventory,'wt',encoding='utf8')as f:
        for key,v in by_inode.items():
            v['category']=min((x['category']for x in v['paths']),key=ORDER.index)
            global_totals[v['category']]+=v['allocated']
            for label in {x['root']for x in v['paths']}:root_totals[label]['allocated_bytes']+=v['allocated'];root_totals[label]['logical_unique_bytes']+=v['logical'];root_totals[label]['unique_files']+=1
            for label in {x['root']for x in v['paths']}:
                c=min((x['category']for x in v['paths']if x['root']==label),key=ORDER.index);root_totals[label][c]+=v['allocated']
            pkeys={}
            for x in v['paths']:
                parts=Path(x['relative']).parts
                if 'parents'not in parts:continue
                i=parts.index('parents')
                if i+1>=len(parts):continue
                # Keep production and published-delivery parent accounts separate.
                pk=(x['root'],str(Path(*parts[:i]))if i else '.',parts[i+1])
                pkeys.setdefault(pk,[]).append(x)
            for pk,aliases in pkeys.items():
                p=parents.setdefault(pk,collections.Counter());c=min((x['category']for x in aliases),key=ORDER.index)
                p['allocated_bytes']+=v['allocated'];p['logical_unique_bytes']+=v['logical'];p[c]+=v['allocated'];p['unique_files']+=1
            eligible=all(x['category']in REBUILDABLE for x in v['paths'])
            if eligible:
                held_outside=v['nlink']>len(v['paths'])
                proposal.append({'paths':[x['path']for x in v['paths']],'category':v['category'],'allocated_bytes_upper_bound':v['allocated'],'nlink':v['nlink'],'links_seen':len(v['paths']),'possible_external_links':held_outside,'release_upper_bound_if_all_aliases_deleted':0 if held_outside else v['allocated'],'status':'proposal_only_owner_approval_required','depends_on':'formal portable export + actual-media readback + downstream intake + retained recipes/evidence/source provenance'})
            f.write(json.dumps({'inode_key':key,**v},ensure_ascii=False)+'\n')
    parent_rows=[]
    for (label,prefix,sid),metrics in sorted(parents.items()):
        root=rootpaths[label]/prefix/'parents'/sid;seal=root/'alpha_seal_completed.json';recipe=root/'input_recipe.json';state='unsealed';members=None;windows=None
        if seal.is_file():
            marker=json.loads(seal.read_text());state=marker['result']['status'];windows=marker['result']['alpha_formal_windows'];members=len(marker['recipe']['input_recipe']['seed_row']['members'])
        elif list(root.glob('failed_*.json')):state='rejected'
        if members is None and recipe.is_file():members=len(json.loads(recipe.read_text())['seed_row']['members'])
        row={'root':label,'prefix':prefix,'scene_id':sid,'state':state,'members':members,'windows':windows,**dict(metrics)};parent_rows.append(row)
    (out/'PARENTS.json').write_text(json.dumps(parent_rows,indent=2))
    (out/'DELETION_PROPOSAL_ONLY.json').write_text(json.dumps(proposal,indent=2))
    result={'status':'measured'if not errors else 'measured_with_scan_errors','roots':{k:str(v)for k,v in rootpaths.items()},'started_unix':start,'completed_unix':time.time(),'elapsed_seconds':time.time()-start,'file_paths':visited,'unique_file_inodes':len(by_inode),'allocated_file_bytes_inode_dedup':sum(v['allocated']for v in by_inode.values()),'allocated_directory_bytes_inode_dedup':sum(dirs.values()),'global_category_allocated_bytes':dict(global_totals),'root_metrics':{k:dict(v)for k,v in root_totals.items()},'parent_accounts':len(parent_rows),'proposal_only_release_upper_bound_bytes':sum(x['release_upper_bound_if_all_aliases_deleted']for x in proposal),'symlinks_not_followed':symlinks,'errors':errors,'files_deleted':0,'caveats':['per-root totals can share inodes; do not add them as global physical occupancy','stat blocks and nlink do not identify shared reflink extents; prospective deletion is an upper bound','AFS is live: measurements have a traversal interval, not an atomic filesystem snapshot','published RIRs and native proofs are retained; no RIR deletion authorization or exact regeneration claim']}
    (out/'SUMMARY.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots-json',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();inspect(json.loads(a.roots_json.read_text()),a.output)
if __name__=='__main__':main()
