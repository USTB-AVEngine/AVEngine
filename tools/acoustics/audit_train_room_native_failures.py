#!/usr/bin/env python3
"""Distinguish strict zero surfaces from positive-area native upload rejects."""
from __future__ import annotations
import argparse,collections,hashlib,io,json,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'src')]
from tools.acoustics.train_house_followup import read,save

def triangle_receipt(vertices,faces):
    counts=collections.Counter();minimum=None;maximum=None
    examples=[]
    for start in range(0,len(faces),200000):
        p=np.asarray(vertices,dtype=np.float32)[faces[start:start+200000]]
        corners=p.astype(np.float64);physical=np.cross(corners[:,1]-corners[:,0],corners[:,2]-corners[:,0]);physical_sq=np.einsum('ij,ij->i',physical,physical)
        native=np.cross((p[:,1]-p[:,0]).astype(np.float64),(p[:,2]-p[:,0]).astype(np.float64));native_sq=np.einsum('ij,ij->i',native,native)
        strict=physical_sq==0;reject=native_sq<=1e-20
        counts.update(dict(triangles=len(p),strict_physical_zero=int(strict.sum()),native_rejected=int(reject.sum()),positive_area_native_rejected=int((reject&~strict).sum()),positive_area_rounded_native_exact_zero=int(((native_sq==0)&~strict).sum()),positive_area_native_cutoff=int((reject&(native_sq>0)&~strict).sum())))
        values=.5*np.sqrt(physical_sq[reject&~strict])
        if len(values):minimum=min(minimum,float(values.min())) if minimum is not None else float(values.min());maximum=max(maximum,float(values.max())) if maximum is not None else float(values.max())
        for i in np.flatnonzero(reject&~strict)[:max(0,5-len(examples))]:examples.append(dict(face_index=int(start+i),vertices_m=corners[i].tolist(),strict_physical_area_m2=float(.5*np.sqrt(physical_sq[i])),native_cross_squared=float(native_sq[i])))
    return dict(counts,minimum_rejected_positive_area_m2=minimum,maximum_rejected_positive_area_m2=maximum,examples=examples)

def exact_package_objects(path):
    """Read immutable, hash-checked native coordinates for diagnostics only.

    Actual acceptance already used the unchanged formal runtime loader. This
    diagnostic does not need to repeat its expensive full source-parity QA.
    """
    path=Path(path);root=path.parent.resolve();m=read(path);arrays={};hashes={}
    for name in ('vertices','triangles'):
        declared=m['arrays'][name];p=(root/declared['path']).resolve()
        if not p.is_relative_to(root):raise ValueError('Diagnostic array escapes package')
        data=p.read_bytes();actual=hashlib.sha256(data).hexdigest()
        if len(data)!=declared['byte_size'] or actual!=declared['sha256']:raise ValueError('Diagnostic array snapshot differs: '+name)
        a=np.load(io.BytesIO(data),allow_pickle=False)
        if tuple(a.shape)!=tuple(declared['shape']) or a.dtype!=np.dtype(declared['dtype']):raise ValueError('Diagnostic array schema differs: '+name)
        arrays[name]=a;hashes[name]=actual
    vertices=np.asarray(arrays['vertices'],dtype=np.float32);triangles=np.asarray(arrays['triangles'],dtype=np.uint32);objects=[]
    for item in m['objects']:
        v=item['vertex_offset'];n=item['vertex_count'];t=item['triangle_offset'];k=item['triangle_count']
        faces=triangles[t:t+k].astype(np.int64)-v
        if faces.min()<0 or faces.max()>=n:raise ValueError('Diagnostic object indices escape vertex partition')
        objects.append(dict(object_id=item['object_id'],vertices=vertices[v:v+n],triangles=faces))
    return objects,dict(manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),array_sha256=hashes,method='immutable NPY byte snapshots verified against declared hashes; identical native float32 coordinates and object partitions; diagnostic only, actual acceptance formal loader unchanged')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--destination',type=Path);a=p.parse_args();w=a.output;rows=[]
    for h in read(w/'inputs/candidates.json'):
        failure=w/'houses'/h['house']/'house_failure.json'
        if not failure.exists():continue
        r=read(failure);reason='; '.join(r['reasons']);detail=[];snapshot=None
        if 'zero-area' in reason or 'POSITIVE_AREA_NATIVE_REJECTION' in reason:
            if h['reuse_package']:
                objects,snapshot=exact_package_objects(h['reuse_package'])
                for obj in objects:
                    metrics=triangle_receipt(obj['vertices'],obj['triangles'])
                    if metrics['native_rejected']:detail.append(dict(object_id=obj['object_id'],**metrics))
            category='positive_area_native_upload_rejected' if sum(d['positive_area_native_rejected'] for d in detail) else 'strict_zero_native_or_unresolved'
        elif h['semantic_conflict']:category='semantic_color_category_conflict'
        else:category='other_native_failure'
        rows.append(dict(house=h['house'],family=h['family'],rooms=h['candidate_room_count'],category=category,reason=reason,objects=detail,source_package=h['reuse_package'],source_snapshot=snapshot,strict_zero_deleted_only=True,positive_area_faces_removed=0,training_eligible=False,judgment='native implementation float32 edge subtraction / squared-cross cutoff rejects retained positive geometry; keep review; no validator or thresholds changed' if category=='positive_area_native_upload_rejected' else 'keep review; original evidence retained'))
        print('NATIVE_FAILURE_AUDIT',h['house'],category,'positive_rejected',sum(d['positive_area_native_rejected'] for d in detail),flush=True)
    save(a.destination or w/'evidence/native_failures_audit.json',dict(rows=rows,categories={c:dict(houses=sum(r['category']==c for r in rows),rooms=sum(r['rooms'] for r in rows if r['category']==c)) for c in sorted({r['category'] for r in rows})},native_check_source=str(REPO/'native/habitat/esp/audio/RLRAcousticContext.cpp')+':503-526',native_check='float32 edge subtraction, then float64 cross squared <= 1e-20',strict_removal_rule='float64 cross on exact stored vertex coordinates is exactly zero; tolerance 0 m2',native_validator_modified=False))
if __name__=='__main__':main()
