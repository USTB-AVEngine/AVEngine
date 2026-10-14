"""Keep immutable witness bytes and paths while sharing identical own copies."""
from pathlib import Path
import hashlib,os


def share_identical_witness_files(directory,cache_directory):
    directory=Path(directory).resolve();cache=Path(cache_directory).resolve()
    if cache==directory or directory in cache.parents:
        raise ValueError('evidence cache must be outside the individual witness directory')
    cache.mkdir(parents=True,exist_ok=True);records=[]
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or path.is_symlink():continue
        if path.name!='native_support_survey.json' and path.suffix not in {'.glb','.navmesh'} and not ('textures' in path.parts and path.suffix in {'.png','.jpg','.jpeg'}):continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest();target=cache/(digest+path.suffix)
        try:os.link(path,target)
        except FileExistsError:pass
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise ValueError('immutable witness cache hash mismatch')
        shared=False
        if not os.path.samefile(path,target):
            temporary=path.with_name(path.name+'.share_'+str(os.getpid()))
            os.link(target,temporary);os.replace(temporary,path);shared=True
        records.append({'path':str(path),'sha256':digest,'size_bytes':path.stat().st_size,'cache_path':str(target),'shared_existing_bytes':shared})
    return {'status':'pass','original_paths_and_bytes_retained':True,'files':records,'shared_existing_files':sum(r['shared_existing_bytes']for r in records)}
