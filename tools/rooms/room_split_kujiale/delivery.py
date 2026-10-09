"""Assemble a compatible draft list, old-room comparison and complete audits."""
from __future__ import annotations
import argparse, copy, csv, json, math, subprocess
from pathlib import Path
from collections import Counter
import shapely
from shapely.geometry import shape
from tools.rooms.room_split_kujiale.adapter import read, dump
from tools.rooms.room_selection.geometry import short_side
from tools.rooms.room_split_kujiale.pipeline import v6_api


def csv_write(path,rows,columns):
    with Path(path).open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items() if k in columns})


def area_bin(a):
    return '6_10' if a<10 else '10_20' if a<20 else '20_30' if a<30 else '30_35'


def assemble(root):
    root=Path(root);delivery=root/'kujiale_delivery_v1/final_v1';plan=read(root/'input_plan_v1.json');quality,_=v6_api()
    files=sorted((delivery/'regions').glob('*.json'));regs=[read(p) for p in files]
    expected={(h['house'],r['room_label']) for h in plan['houses'] for r in h['rooms']}
    if len(regs)!=len(expected) or {(r['house'],r['source_region']) for r in regs}!=expected:raise ValueError('Missing or duplicate input source rooms')
    rooms=[b for r in regs for b in r['blocks']];kept=[b for b in rooms if b['decision']=='retain'];pending=[b for b in rooms if b['decision']=='unresolved']
    roomids=[b['id'] for b in rooms]
    if len(roomids)!=len(set(roomids)) or {p.stem for p in (delivery/'rooms').glob('*.json')}!=set(roomids):
        raise ValueError('Duplicate, missing or extra final room files')
    oldkeys={tuple(x) for x in plan['old_production_keys']}
    columns=['id','house','source_region','source_region_id','floor_id','floor_y_m','floor_area_m2','short_side_m','black_fraction','source','origin_list','source_selection','floor_polygon_xz_m','placement_witness','leakage']
    draft=[]
    for b in kept:
        row={k:b.get(k) for k in columns};row.update(source='cut' if b.get('new_room') else 'original',
            origin_list=['old_v7_S_production'] if (b['house'],b['source_region']) in oldkeys else ['new_geometry_candidate'],
            source_selection=str(delivery/'rooms'/(b['id']+'.json')),decision='retain',room_type=b['room_type'],
            black_fraction_status='not_applicable_CAD',leakage=None,leakage_status='pending_other_task',
            leakage_policy={'test_max':.05,'train_max':.15,'above_train_max':'exclude'},production_integrated=False)
        draft.append(row)
    comparisons=[];source_outcomes=[];large=[];errors=[];shape_counts=Counter();activecuts=[]
    maximum_partition_symmetric_difference_m2=0.;maximum_partition_overlap_m2=0.
    required=('id','house','source_region','floor_id','floor_y_m','floor_polygon_xz_m','floor_area_m2','short_side_m','decision','placement_witness')
    for reg in regs:
        key=(reg['house'],reg['source_region']);live=[b for b in reg['blocks'] if b['decision']=='retain'];unres=[b for b in reg['blocks'] if b['decision']=='unresolved'];drop=[b for b in reg['blocks'] if b['decision']=='discard']
        original=shape(reg['source_geometry']['floor_polygon_xz_m'])
        unchanged=len(live)==1 and shape(live[0]['floor_polygon_xz_m']).symmetric_difference(original).area<=1e-6
        outcome='retained_unchanged' if unchanged else 'retained_changed_or_split' if live else 'unresolved' if unres else 'discarded'
        if live and unres:outcome='partially_unresolved'
        row=dict(house=key[0],source_region=key[1],source_room_type=reg['source_geometry']['source_room_type'],room_type=reg['source_geometry']['room_type'],
            old_production_admitted=key in oldkeys,old_floor_area_m2=float(original.area),requires_split=reg['requires_split'],
            outcome=outcome,retained_rooms=len(live),retained_ids=[b['id'] for b in live],retained_areas_m2=[b['floor_area_m2'] for b in live],
            retained_area_m2=sum(b['floor_area_m2'] for b in live),discarded_blocks=len(drop),discarded_area_m2=sum(b['floor_area_m2'] for b in drop),
            unresolved_blocks=len(unres),unresolved_area_m2=sum(b['floor_area_m2'] for b in unres),
            reasons=sorted({x for b in drop+unres for x in b.get('discard_reasons',[])+b.get('unresolved_reasons',[])}),
            source_file=str(delivery/'regions'/(key[0]+'__'+key[1]+'.json')),old_production_source=plan['old_production_source'])
        row['change']='unchanged' if unchanged else 'split' if len(live)>1 else 'shape_changed' if live else 'unresolved' if unres else 'dropped'
        source_outcomes.append(row)
        if key in oldkeys:comparisons.append(row)
        if reg['requires_split']:large.append(row)
        part_error=abs(sum(b['floor_area_m2'] for b in reg['blocks'])-original.area)
        if part_error>1e-5:errors.append(dict(region=list(key),kind='area_partition',error_m2=part_error))
        partition=shapely.union_all([shape(b['floor_polygon_xz_m']) for b in reg['blocks']])
        difference=partition.symmetric_difference(original).area
        overlap=max(0.,sum(b['floor_area_m2'] for b in reg['blocks'])-partition.area)
        maximum_partition_symmetric_difference_m2=max(maximum_partition_symmetric_difference_m2,difference)
        maximum_partition_overlap_m2=max(maximum_partition_overlap_m2,overlap)
        if difference>1e-5:errors.append(dict(region=list(key),kind='partition_gaps_or_escape',area_m2=difference))
        if overlap>1e-5:errors.append(dict(region=list(key),kind='overlapping_partition_blocks',area_m2=overlap))
        retained_shapes=[]
        for b in reg['blocks']:
            if any(k not in b for k in required):errors.append(dict(room=b['id'],kind='missing_fields'))
            g=shape(b['floor_polygon_xz_m'])
            file=delivery/'rooms'/(b['id']+'.json')
            if not file.is_file() or read(file)!=b:errors.append(dict(room=b['id'],kind='room_region_parity'))
            if abs(g.area-b['floor_area_m2'])>1e-6 or abs(short_side(g)-b['short_side_m'])>1e-6:errors.append(dict(room=b['id'],kind='geometry_numbers'))
            if g.difference(original.buffer(1e-7)).area>1e-5:errors.append(dict(room=b['id'],kind='escaped_source_floor'))
            if b['decision']!='retain':continue
            retained_shapes.append(g)
            if not 6-1e-7<=g.area<=35+1e-7 or short_side(g)<2.4-1e-7:errors.append(dict(room=b['id'],kind='area_or_short_side'))
            w=b['placement_witness']
            if not w.get('found') or w.get('validation',{}).get('status')!='pass':errors.append(dict(room=b['id'],kind='missing_validated_witness'))
            if not b.get('connectivity',{}).get('pass_'):errors.append(dict(room=b['id'],kind='connectivity'))
            dd=quality.defects(g);shape_counts['NECK']+=dd['neck_count'];shape_counts['CORRIDOR']+=dd['corridor_count']
        shape_counts['WRAP']+=quality.wrap_count(retained_shapes)
        for c in reg['cut_lines']:
            if not c.get('active_in_final_partition',True):continue
            activecuts.append(c)
            shape_counts['FURNITURE']+=int(c.get('furniture_intersection_length_m',0)>.5)
            if c.get('segment_count',0)>3:errors.append(dict(cut=c['id'],kind='cut_segments'))
            if c.get('angle_fallback_deg',0)>15+1e-6:errors.append(dict(cut=c['id'],kind='angle_fallback'))
    bins=dict(Counter(area_bin(b['floor_area_m2']) for b in kept))
    for k in ('6_10','10_20','20_30','30_35'):bins.setdefault(k,0)
    all_counts={k:int(shape_counts[k]) for k in ('NECK','CORRIDOR','WRAP','FURNITURE')}
    validation=dict(maximum_partition_symmetric_difference_m2=maximum_partition_symmetric_difference_m2,
         maximum_partition_overlap_m2=maximum_partition_overlap_m2,source_rooms=len(regs),retained_rooms=len(kept),errors=errors,all_room_shape_counts=all_counts,
         validated_placement_witnesses=sum(b['placement_witness'].get('validation',{}).get('status')=='pass' for b in kept),
         all_room_scope=True,independent_claude_checkers_scope='requires_split sources; supplementary all-source audit view also supplied',
         active_cut_lines=len(activecuts),cut_segment_hist=dict(Counter(c['segment_count'] for c in activecuts)),
         max_cut_furniture_m=max((c['furniture_intersection_length_m'] for c in activecuts),default=0),
         status='pass' if not errors and not any(all_counts.values()) else 'fail')
    dump(root/'evidence/all_room_delivery_validation_v1.json',validation)
    summary=dict(total_source_rooms=len(regs),source_houses=plan['source_houses'],source_outcomes=dict(Counter(r['outcome'] for r in source_outcomes)),
        total_rooms=len(kept),total_area_m2=sum(b['floor_area_m2'] for b in kept),area_bins=bins,
        split_large_living_sources=len(large),large_living_retained_rooms=sum(r['retained_rooms'] for r in large),
        large_living_discarded_blocks=sum(r['discarded_blocks'] for r in large),large_living_discarded_area_m2=sum(r['discarded_area_m2'] for r in large),
        large_living_unresolved_blocks=sum(r['unresolved_blocks'] for r in large),
        discarded_blocks=sum(b['decision']=='discard' for b in rooms),discarded_area_m2=sum(b['floor_area_m2'] for b in rooms if b['decision']=='discard'),
        pending=len(pending),unresolved_source_regions=sum(bool(r['unresolved_blocks']) for r in source_outcomes),
        old_admitted_rooms=len(oldkeys),old_admitted_changes=dict(Counter(r['change'] for r in comparisons)),
        new_source_candidates=sum(r['retained_rooms']>0 and not r['old_production_admitted'] for r in source_outcomes),
        new_rooms_from_previously_unadmitted_sources=sum(r['retained_rooms'] for r in source_outcomes if not r['old_production_admitted']),
        lists_modified=False,production_integrated=False,acoustics='not_run',leakage_columns_empty=True,
        black_fraction_status='not_applicable_CAD',max_room_area_m2=35,min_room_area_m2=6,min_short_side_m=2.4,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_navmesh_map=str(root/'navmesh_map_v1.json'),sources=[str(delivery)],validation=validation)
    dump(delivery/'summary.json',summary)
    draftdir=root/'kujiale_room_list_draft_v1';draftdir.mkdir(exist_ok=True)
    dump(draftdir/'room_list_draft_v1.json',dict(schema='Kujiale_room_list_draft_v1_cap35',rooms=draft,leakage='pending_other_task',production_integrated=False))
    dump(draftdir/'summary.json',summary);dump(draftdir/'pending.json',pending)
    csv_write(draftdir/'rooms.csv',draft,columns)
    csv_write(root/'comparison_old_33_rooms_v1.csv',comparisons,list(comparisons[0]))
    csv_write(root/'source_outcomes_151_v1.csv',source_outcomes,list(source_outcomes[0]))
    csv_write(root/'large_living_14_v1.csv',large,list(large[0]))
    dump(root/'comparison_old_33_rooms_v1.json',dict(source=plan['old_production_source'],old_admitted=comparisons,new_sources=[r for r in source_outcomes if r['retained_rooms'] and not r['old_production_admitted']],production_modified=False))
    # This read-only derived view permits the owner's split-only scripts to audit
    # every retained CAD room and draw every source with the identical renderer.
    view=root/'all_source_audit_view_v1'
    for file,reg in zip(files,regs):
        audit=copy.deepcopy(reg);audit['requires_split']=True;audit['audit_view_only']=True;audit['actual_requires_split']=reg['requires_split']
        dump(view/'regions'/file.name,audit)
    dump(view/'README.json',dict(scope='all 151 sources; requires_split true solely to select all rows in independent audit and redraw scripts; actual delivery unchanged',delivery=str(delivery)))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if validation['status']!='pass':raise RuntimeError('Delivery validation failed')

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',required=True);v=a.parse_args();assemble(v.root)
