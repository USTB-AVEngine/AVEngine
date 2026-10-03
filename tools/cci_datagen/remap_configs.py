#!/usr/bin/env python3
"""Write new path-mapped runtime configs; never rewrite copied raw manifests.

Explicit input list only. Compiled package manifests/QA must remain immutable.
A mapping receipt commits both source and effective configs, and reports missing
referenced asset paths. Accepted CCI requests must select the effective configs.
"""
import argparse
import json
from pathlib import Path
import os
from generate import sha,contained
from resources import write_json

def mapped(value,mirror,copies):
    if isinstance(value,dict):return {k:mapped(v,mirror,copies) for k,v in value.items()}
    if isinstance(value,list):return [mapped(v,mirror,copies) for v in value]
    if isinstance(value,str) and value.startswith('/data/'):
        base,separator,fragment=value.partition('#')
        return str(copies.get(base,Path(mirror)/base.lstrip('/')))+(separator+fragment if separator else '')
    return value

def paths(value):
    if isinstance(value,dict):
        for v in value.values():yield from paths(v)
    elif isinstance(value,list):
        for v in value:yield from paths(v)
    elif isinstance(value,str) and value.startswith('/mnt/afs/'):yield value

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-jsons',required=True,help='JSON array of 48g absolute config paths, already mirrored')
    p.add_argument('--mirror-root',required=True)
    p.add_argument('--output-root',required=True)
    a=p.parse_args();root=Path(a.output_root).resolve();mirror=Path(a.mirror_root).resolve()
    if os.environ.get('CCI_DATAGEN_BASE'):contained(root,os.environ['CCI_DATAGEN_BASE'])
    sources=json.loads(Path(a.source_jsons).read_text())
    if not sources or len(set(sources))!=len(sources):raise ValueError('explicit unique runtime config list is required')
    for source in sources:
        if not source.startswith('/data/') or '/package/' in source or '/qa/' in source or Path(source).name in ('manifest.json','catalog.json'):
            raise ValueError('compiled manifests, package QA and room catalogs are not runtime path-map inputs')
    copies={s:root/s.lstrip('/') for s in sources};rows=[];root.mkdir(parents=True,exist_ok=False)
    for source in sources:
        old=mirror/source.lstrip('/');new=copies[source];data=mapped(json.loads(old.read_text()),mirror,copies)
        write_json(new,data)
        rows.append({'source':str(old),'source_sha256':sha(old),'effective':str(new),'effective_sha256':sha(new)})
    for row in rows:
        checked=set(paths(json.loads(Path(row['effective']).read_text())))
        row['referenced_paths_checked']=len(checked)
        row['missing_referenced_paths']=[s for s in sorted(checked) if not Path(s.split('#',1)[0]).exists()]
    write_json(root/'PATH_MAPPING.json',{'schema':'cci_runtime_path_mapping_v1','rows':rows,'compiled_manifests_modified':False})
    print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':main()
