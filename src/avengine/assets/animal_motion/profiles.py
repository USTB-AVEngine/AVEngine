"""Resolve target rigs from source metadata; never substitute a donor mesh."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from avengine.assets.roster_motion.io import read_glb
from avengine.assets.roster_motion.pose import accessor, globals_at, pose, animation_times, skin_influences
from avengine.assets.roster_motion.contact import subtree, find_bone

PAWS = ('FL', 'FR', 'HL', 'HR')
ANCHORS = ('paw_front_left', 'paw_front_right', 'paw_hind_left', 'paw_hind_right')

@dataclass
class Rig:
    asset_id: str
    species: str
    path: Path
    document: dict
    binary: bytes
    basis: np.ndarray
    head: int
    tail: int
    body: int
    mouth_bone: int
    mouth_local: np.ndarray
    chains: dict
    masks: list
    head_masks: list
    provenance: dict

    @classmethod
    def load(cls, record):
        files = record['files']
        key = next((k for k in ('motion_master_glb', 'motion_runtime_glb', 'walking_candidate_master') if k in files), None)
        if key is None:
            raise ValueError('animation_missing:no_skin_or_walking_source')
        path = Path(files[key]['path'])
        d, b = read_glb(path)
        if not d.get('skins') or not any(a.get('name') == 'Walking' for a in d.get('animations', [])):
            raise ValueError('animation_missing:Walking_or_skin')
        names = {n.get('name'): i for i, n in enumerate(d['nodes'])}
        parents = {j:i for i,n in enumerate(d['nodes']) for j in n.get('children', [])}
        basis = np.eye(4)
        if record.get('walking_candidate'):
            candidate = json.loads(Path(record['walking_candidate']).read_text())
            manifest = json.loads(Path(candidate['source_package_manifest']).read_text())
            anchors = {a['anchor_id']: names[a['joint_id']] for a in manifest['anchors']}
            scale = np.asarray(candidate['scale_xyz'], float)
            if np.ptp(scale) > 1e-6:
                raise ValueError('nonuniform_target_scale_requires_separate_rig_qualification')
            basis[:3,:3] = np.diag(scale) @ np.asarray(candidate['rotation'])
            basis[:3,3] = scale * np.asarray(candidate['translation_m'])
            head = anchors['head']; body = anchors['body']
            mouth_bone = head; mouth_local = np.asarray(candidate['emitter_head_local_m'])
            chains = {}
            for side, anchor in zip(PAWS, ANCHORS):
                toe = anchors[anchor]
                chain = [toe]
                for _ in range(3):
                    if chain[-1] not in parents:
                        raise ValueError('semantic_paw_ancestry_too_short:' + side)
                    chain.append(parents[chain[-1]])
                chains[side] = list(reversed(chain))
            # Tail is the rear-most central joint in this target, excluding
            # all mapped limb and head subtrees. Save the chosen node identity.
            excluded = set(subtree(d, head))
            for chain in chains.values(): excluded.update(subtree(d, chain[0]))
            idle = next(a for a in d['animations'] if a['name']=='Idle')
            g = {j:basis@m for j,m in globals_at(d,b,idle,0).items()}
            joints = set(j for s in d['skins'] for j in s['joints'])
            eligible = [j for j in joints-excluded if g[j][1,3] > min(g[c[2]][1,3] for c in chains.values()) + .02]
            if not eligible:raise ValueError('target_tail_landmark_unresolved')
            tail = max(eligible, key=lambda j:g[j][2,3])
            provenance = {'candidate':record['walking_candidate'], 'manifest':candidate['source_package_manifest'],
                'tail_definition':'rear-most own non-limb non-head skeletal joint above paws, in registered normalization',
                'source_basis':basis.tolist()}
        else:
            head=find_bone(d,' Head');body=find_bone(d,' Pelvis');tail=find_bone(d,' Tail')
            mouth_bone=find_bone(d,'Xtra Mouth');mouth_local=np.zeros(3)
            chains={s:[find_bone(d,' '+side+' '+p) for p in parts] for s,side,parts in (
                ('FL','L',('UpperArm','Forearm','Hand','Finger0')),('FR','R',('UpperArm','Forearm','Hand','Finger0')),
                ('HL','L',('Thigh','Calf','Foot','Toe0')),('HR','R',('Thigh','Calf','Foot','Toe0')))}
            provenance={'tail_definition':'original named Tail root','source_basis':basis.tolist()}
        masks=[];head_masks=[]
        for node in d['nodes']:
            if 'mesh' not in node:continue
            if 'skin' not in node:raise ValueError('unskinned_part_in_motion_master')
            joints=d['skins'][node['skin']]['joints']
            for primitive in d['meshes'][node['mesh']]['primitives']:
                js,ws=skin_influences(d,b,primitive['attributes'])
                def mask(nodes):return (np.isin(js,[k for k,j in enumerate(joints) if j in nodes])*ws).sum(1)>.15
                masks.append({s:mask(set(subtree(d,c[2]))) for s,c in chains.items()})
                head_masks.append(mask(set(subtree(d,head))))
        if any(not any(m[s].any() for m in masks) for s in PAWS):raise ValueError('target_semantic_paw_skin_missing')
        provenance.update(head_node={'id':head,'name':d['nodes'][head].get('name')},tail_node={'id':tail,'name':d['nodes'][tail].get('name')},
            chains={s:[{'id':i,'name':d['nodes'][i].get('name')} for i in c] for s,c in chains.items()})
        return cls(record['asset_id'],record['species'],path,d,b,basis,head,tail,body,mouth_bone,mouth_local,chains,masks,head_masks,provenance)

    def action(self,name='Walking'):
        return next(a for a in self.document['animations'] if a['name']==name)

    def globals(self,time_s,action='Walking'):
        return {j:self.basis@m for j,m in globals_at(self.document,self.binary,self.action(action),float(time_s)).items()}

    def enable_skin_cache(self):
        from .skin_cache import CachedSkin
        self._skin_cache=CachedSkin(self.document,self.binary)
        return self

    def skin(self,g):
        cached=getattr(self,'_skin_cache',None)
        if cached is not None:return cached(g)
        return pose(self.document,self.binary,None,globals_override=g,canonical_flip=False,ground=False)[0]

    def landmarks(self,g):
        return {'head':g[self.head][:3,3].copy(), 'tail':g[self.tail][:3,3].copy(),
                'mouth':(g[self.mouth_bone]@np.r_[self.mouth_local,1.])[:3], 'body':g[self.body][:3,3].copy()}

    def samples(self,name='Walking'):
        return animation_times(self.document,self.binary,self.action(name))


def paw_points(parts,masks):
    result={}
    for side in PAWS:
        choices=[]
        for pi,(p,m) in enumerate(zip(parts,masks)):
            ids=np.flatnonzero(m[side])
            if len(ids):
                vi=int(ids[np.argmin(p['positions'][ids,1])]);choices.append((float(p['positions'][vi,1]),pi,vi))
        _,pi,vi=min(choices);result[side]={'part':pi,'vertex':vi,'point':parts[pi]['positions'][vi].copy()}
    return result
