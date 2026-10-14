"""Prepare a deficit-driven batch using original registries; render/export nothing."""
from pathlib import Path
import argparse,json
from avengine.timeline.v7_production_inputs import attach_runtime_contract
from avengine.timeline.v7_demand import read_demand_state,plan_phase2_demand,DEMAND_REVISION


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('seeds','observed-state','assets','source-admission','contract','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--focus',choices=['binding','human_motion','animal_motion','count','occlusion','taxonomy','content_binding'])
    p.add_argument('--offset',type=int,default=0);p.add_argument('--limit',type=int,default=4)
    p.add_argument('--scene-prefix',required=True,help='Fresh task-owned parent IDs; original recipes remain read-only')
    a=p.parse_args()
    if a.output.exists():raise ValueError('use a fresh demand batch filename')
    if a.offset<0 or a.limit<1:raise ValueError('invalid batch bounds')
    catalog=json.loads(a.assets.read_text());catalog=catalog['assets'] if isinstance(catalog,dict) else catalog
    source=json.loads(a.source_admission.read_text());contract=json.loads(a.contract.read_text())
    state=read_demand_state(a.observed_state)
    rows=[json.loads(line) for line in a.seeds.read_text().splitlines() if line.strip()][a.offset:a.offset+a.limit]
    if not rows:raise ValueError('empty demand batch')
    planned=[]
    for i,row in enumerate(rows):
        original=row['scene_id'];row['scene_id']=a.scene_prefix+f'_{i:03d}';row['independent_parent_scene_id']=row['scene_id']
        row['demand_source_recipe_scene_id']=original
        row['demand_source_production_sampling']=row.get('production_sampling')
        row.pop('production_sampling',None)
        row=attach_runtime_contract(row,{'production_input_contract':str(a.contract.resolve())})
        row=plan_phase2_demand(row,state,catalog,source['records'],contract,focus=a.focus)
        planned.append(row)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:
        for row in planned:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps({'revision':DEMAND_REVISION,'parents':len(planned),'output':str(a.output),
        'focuses':[r['scene_demand']['focus'] for r in planned],
        'requested_motion':[r.get('motion_requests',{}) for r in planned],
        'training_table_exported':False,'native_jobs_started':False},ensure_ascii=False))


if __name__=='__main__':main()
