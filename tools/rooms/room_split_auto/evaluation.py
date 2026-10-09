"""House-disjoint tune selection and one recorded holdout IoU evaluation."""
from pathlib import Path
from collections import defaultdict,Counter
import json
import numpy as np
import shapely
from shapely.geometry import shape,box
from scipy.optimize import linear_sum_assignment
from .pipeline import dump,now


def load_regions(root,run_name,param_id):
    result={}
    for path in sorted((Path(root)/run_name/param_id).glob('*.json')):
        for r in json.loads(path.read_text()).get('regions',[]):result[(r['house'],r['source_region'])]=r
    return result


def score_reference_pair(region,refs,retained_only=False):
    """Only manual kept boxes are targets; unused auto blocks are reported separately."""
    blocks=[b for b in region.get('blocks',[]) if b['decision']!='unresolved' and (not retained_only or b['decision'] in ('retain','unchanged'))]
    manuals=[]
    floors=region['source_geometry']['floors']
    for ref in refs:
        d=json.loads(Path(ref['path']).read_text());ys=d.get('floor_y_m')
        matching=[shape(f['floor_polygon']) for f in floors if ys is None or abs(f['floor_y_m']-ys)<=0.3]
        geometry=shapely.union_all(matching).intersection(box(*d['bbox_xz_m'][0],*d['bbox_xz_m'][1])) if matching else box(0,0,0,0)
        manuals.append((d,geometry))
    matrix=np.zeros((len(refs),len(blocks)))
    for i,(d,g) in enumerate(manuals):
        for j,b in enumerate(blocks):
            if d.get('floor_y_m') is not None and abs(d['floor_y_m']-b['floor_y_m'])>0.3:continue
            automatic=shape(b['floor_polygon_xz_m']);den=automatic.union(g).area
            matrix[i,j]=automatic.intersection(g).area/den if den else 0.0
    matched={}
    if len(blocks):
        ii,jj=linear_sum_assignment(matrix,maximize=True);matched=dict(zip(ii,jj))
    entries=[]
    for i,(d,g) in enumerate(manuals):
        j=matched.get(i)
        entries.append({'manual_id':d['id'],'manual_json':refs[i]['path'],'manual_floor_area_m2':float(g.area),'manual_bbox_area_m2':float(box(*d['bbox_xz_m'][0],*d['bbox_xz_m'][1]).area),'manual_geometry_status':'source_region_semantic_floor_clipped_bbox' if not g.is_empty else 'unresolved_no_matching_structural_floor','matched_auto_id':blocks[j]['id'] if j is not None else None,'matched_auto_decision':blocks[j]['decision'] if j is not None else None,'matched_iou':float(matrix[i,j]) if j is not None else 0.0,'best_iou':float(matrix[i].max()) if len(blocks) else 0.0,'manual_scope_basis':d.get('scope_basis'),'reference_short_side_below_2_4':bool(min(d['extent_m'])<2.4) if d.get('extent_m') else None})
    return {'house':region['house'],'source_region':region['source_region'],'requires_split':region.get('requires_split'),'source_area_m2':region.get('source_floor_area_m2'),'source_status':region['status'],'manual_count':len(entries),'automatic_block_count':len(blocks),'extra_automatic_blocks':len(blocks)-len(matched),'source_mean_iou':float(np.mean([e['matched_iou'] for e in entries])) if entries else 0.0,'entries':entries,'discard_reasons':dict(Counter(x for b in region.get('blocks',[]) for x in b.get('discard_reasons',[]))),'source_unresolved_reasons':region.get('unresolved_reasons',[])}


def score(root,regions,fold):
    split=json.loads((Path(root)/'house_hash_split_v1.json').read_text());houses={h['house'] for h in split['houses'] if h['fold']==fold}
    groups=defaultdict(list);excluded=[]
    for ref in split['reference_rooms']:
        if ref['house'] not in houses:continue
        if ref['id'] in split['dagger_ids']:
            excluded.append(dict(ref,reason='dagger_original_bbox_not_narrowed; excluded by task'));continue
        groups[(ref['house'],ref['source_room_label'])].append(ref)
    rows=[];retained=[]
    for key,refs in sorted(groups.items()):
        r=regions.get(key)
        if r is None:
            rows.append({'house':key[0],'source_region':key[1],'source_mean_iou':0.0,'manual_count':len(refs),'entries':[{'manual_id':a['id'],'matched_iou':0.0,'reason':'AUTO_REGION_MISSING'} for a in refs]});continue
        rows.append(score_reference_pair(r,refs));retained.append(score_reference_pair(r,refs,True))
    return {'fold':fold,'houses':len(houses),'source_rooms_total':sum(r['house'] in houses for r in split['source_rooms']),'scorable_source_rooms':len(rows),'manual_rooms_scored':sum(r['manual_count'] for r in rows),'dagger_rooms_excluded':len(excluded),'dagger_entries':excluded,'mean_iou_source_macro':float(np.mean([r['source_mean_iou'] for r in rows])) if rows else None,'mean_iou_manual_micro':float(np.mean([e['matched_iou'] for r in rows for e in r['entries']])) if rows else None,'retained_only_mean_iou_source_macro':float(np.mean([r['source_mean_iou'] for r in retained])) if retained else None,'source_rows':rows,'retained_only_rows':retained,'target':0.7,'target_met':bool(rows and np.mean([r['source_mean_iou'] for r in rows])>=0.7),'iou_area_basis':'unrounded same-height source-semantic-ground union clipped by reference bbox; no furniture subtraction; rectangular/nav area never substitutes','unmatched_or_unresolved_score':0.0}


def select(root,run_name):
    root=Path(root);grid=json.loads((root/'parameter_search_plan_v1.json').read_text())['grid'];candidates=[]
    for param in grid:
        result=score(root,load_regions(root,run_name,param['id']),'tune');dump(root/(run_name+'_scores')/(param['id']+'.json'),result)
        distance=abs(param['width_max']-1.6)+abs(param['coverage']-0.8)+abs(param['max_distance']-6)/10
        candidates.append({'parameter':param,'mean_iou_source_macro':result['mean_iou_source_macro'],'mean_iou_manual_micro':result['mean_iou_manual_micro'],'default_distance_tiebreak':distance,'score_file':str(root/(run_name+'_scores')/(param['id']+'.json'))})
    chosen=sorted(candidates,key=lambda r:(-r['mean_iou_source_macro'],r['default_distance_tiebreak'],r['parameter']['id']))[0]
    lock={'selected_at_utc':now(),'selected_parameter':chosen['parameter'],'selected_from':'tune houses only','tune_mean_iou_source_macro':chosen['mean_iou_source_macro'],'selection_plan':str(root/'parameter_search_plan_v1.json'),'candidates':candidates,'holdout_evaluation_count_at_selection':0}
    dump(root/'selected_parameter_v1.json',lock)
    with (root/'PROGRESS_zh.md').open('a') as f:f.write('\n## 仅 tune 选择参数\n'+json.dumps(chosen,ensure_ascii=False,indent=2)+'\n留出尚未评分；参数已锁定 selected_parameter_v1.json。\n')
    print(json.dumps(chosen,ensure_ascii=False,indent=2))
    return lock


def evaluate(root,run_name,fold):
    root=Path(root);lock=json.loads((root/'selected_parameter_v1.json').read_text());pid=lock['selected_parameter']['id'];regions=load_regions(root,run_name,pid)
    if fold=='holdout':
        receipt=root/'holdout_evaluation_once.json'
        if receipt.exists():raise FileExistsError('Holdout has already been evaluated; no second evaluation allowed')
        # Claim the single evaluation before computing any score, including failures.
        dump(receipt,{'started_at_utc':now(),'run_name':run_name,'parameter':lock['selected_parameter'],'evaluation_ordinal':1,'status':'started; result stored separately'})
        result=score(root,regions,'holdout');dump(root/'holdout_scores_v1.json',result)
        with (root/'PROGRESS_zh.md').open('a') as f:f.write('\n## 留出唯一一次评分\n'+json.dumps({k:v for k,v in result.items() if k not in ('source_rows','retained_only_rows','dagger_entries')},ensure_ascii=False,indent=2)+'\n出处 holdout_scores_v1.json；不得用留出结果再选择参数。\n')
    else:
        result=score(root,regions,'tune');dump(root/'tune_selected_scores_v1.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_rows','retained_only_rows','dagger_entries')},ensure_ascii=False,indent=2));return result
