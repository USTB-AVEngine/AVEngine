#!/usr/bin/env python3
"""Resume registered content/selection in a fresh attempt with zero native renders."""
from pathlib import Path
import argparse,hashlib,json,os,shutil
from avengine.timeline.v7_native_stage_resume import reuse_native_parent,_link_tree,require_completed_registered_media
from avengine.timeline.v7_registered_generation import generate_registered_parent
from avengine.timeline.v7_demand import SELECTION_FEEDBACK_ENV


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--source-receipt',type=Path,required=True);p.add_argument('--output-native',type=Path,required=True);a=p.parse_args()
    m=json.loads(a.manifest.read_text());owned=Path(m['output_root']).resolve();record=json.loads(a.source_receipt.read_text())
    if not a.source_receipt.resolve().is_relative_to(owned)or record.get('status')!='failed':raise ValueError('only a stopped, failed caller-owned attempt may resume')
    source=owned/'native/parents'/record['scene_id'];recipe=json.loads((source/'input_recipe.json').read_text());row=recipe['seed_row'];output=a.output_native.resolve();target=output/'parents'/row['scene_id']
    if not output.is_relative_to(owned)or target.exists():raise ValueError('fresh owned output-native directory required')
    require_completed_registered_media(source,row)
    provenance=reuse_native_parent(source.parent.parent,output,row)
    shutil.copyfile(source/'input_recipe.json',target/'input_recipe.json')
    # Completed QA pools are immutable inputs from this same physical attempt.
    # Their original recipes still require exact matching in the producer.
    copied=[]
    for member in row['members']:
        for prefix in ('windows_','frozen_dialogue_pools_'):
            marker=source/(prefix+member+'_completed.json')
            if not marker.exists():continue
            saved=json.loads(marker.read_text());_link_tree(source/saved['attempt'],target/saved['attempt']);shutil.copyfile(marker,target/marker.name);copied.append(marker.name)
    runtime=recipe['runtime_inputs'];catalog=json.loads(Path(m['assets_cpu']).read_text());catalog=catalog['assets']if isinstance(catalog,dict)else catalog
    contract=json.loads(Path(runtime['production_input_contract']).read_text());admission=json.loads(Path(contract['inputs']['source_admission']['path']).read_text());human=json.loads(Path(m['human_registered_cpu']).read_text())
    os.environ[SELECTION_FEEDBACK_ENV]=str(owned/'inputs/demand_state.json')
    result=generate_registered_parent(row,recipe['house'],runtime,catalog,admission,human,str(output),resume_only=True)
    source_sha=hashlib.sha256((source/'input_recipe.json').read_bytes()).hexdigest()
    provenance.update(source_input_recipe_sha256=source_sha,completed_question_pools_reused=copied,answers_or_actual_media_modified=False,new_native_renders=0)
    (target/'registered_recovery_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({**result,'new_native_renders':0,'native_question_recovery':{'source_parent':str(source)},'registered_recovery':provenance,'parent_directory':str(target),'seal_path':str(target/'alpha_seal_completed.json')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
