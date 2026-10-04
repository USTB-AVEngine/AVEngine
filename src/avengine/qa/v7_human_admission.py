"""Automatic human skin/contact and retained-license receipts; originals are read-only."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np

def skin_native_vertices(compiled, joint_world_matrices):
    matrices=np.asarray(joint_world_matrices,float) @ compiled.inverse_bind_matrices
    outputs=[];weights=[]
    for p in compiled.primitives:
        transformed=np.einsum("vjab,vb->vja",matrices[p.joint_ordinals],np.c_[p.positions,np.ones(len(p.positions))])
        outputs.append(np.sum(transformed[:,:,:3]*p.weights[:,:,None],axis=1))
        weights.append((p.joint_ordinals,p.weights))
    vertices=np.concatenate(outputs)
    if not np.isfinite(vertices).all():raise ValueError("nonfinite native skin")
    return vertices,weights

def native_sole_measurements(vertices, weights, joint_names, floor_y):
    soles=[]
    for side in ("L","R"):
        names=[i for i,n in enumerate(joint_names) if n.startswith("Bip01 "+side+" Foot") or n.startswith("Bip01 "+side+" Toe")]
        if not names:raise ValueError("human foot joints missing")
        mask=np.concatenate([np.sum(w*np.isin(j,names),axis=1)>=.5 for j,w in weights])
        if not mask.any():raise ValueError("no foot skin vertices")
        soles.append(float(np.min(vertices[mask,1])-floor_y))
    return dict(left_sole_y_relative_m=soles[0],right_sole_y_relative_m=soles[1],
                body_min_y_relative_m=float(vertices[:,1].min()-floor_y))

def qualify_action_samples(samples, *, maximum_contact_error_m, proxy_covers_pose):
    good=[min(abs(s["left_sole_y_relative_m"]),abs(s["right_sole_y_relative_m"]))<=maximum_contact_error_m
          and s["body_min_y_relative_m"]>=-maximum_contact_error_m for s in samples]
    ok=bool(samples) and all(good) and proxy_covers_pose
    return {"status":"pass" if ok else "blocked","native_sample_count":len(samples),"grounded_samples":sum(good),
            "maximum_closest_sole_error_m":max([min(abs(s["left_sole_y_relative_m"]),abs(s["right_sole_y_relative_m"])) for s in samples],default=0.),
            "maximum_floor_penetration_m":max([0.]+[-s["body_min_y_relative_m"] for s in samples]),
            "contact_gate_m":maximum_contact_error_m,"proxy_covers_all_native_skin_vertices":bool(proxy_covers_pose),
            "reason":None if ok else "skin_outside_proxy" if not proxy_covers_pose else "foot_floating_or_floor_penetration"}

def bind_retained_license(path, retained):
    source=Path(path);target=Path(retained);text=source.read_text()
    if "MIT License" not in text or "Copyright (c) 2020 Microsoft" not in text:raise ValueError("unexpected license")
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x") as f:f.write(text)
    return {"status":"pass","spdx":"MIT","upstream_path":str(source),"retained_text":str(target),
            "sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"condition":"retain copyright and permission notice"}
