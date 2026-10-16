"""Validate registered evidence against actual GLBs, native measurements and owner scope."""
import argparse,base64,collections,hashlib,io,json,math
from pathlib import Path
from html.parser import HTMLParser
import numpy as np
from PIL import Image
from v7_object_prepare import dump
from v7_object_color import measure as measure_color, summarize_views
from measure_static_upright_correction import load_glb
from v7_object_progress import append as progress
from v7_object_finish_repair import geometry_match

class ImageAudit(HTMLParser):
    def __init__(self):
        super().__init__()
        self.count=0
        self.errors=[]
    def handle_starttag(self,tag,attrs):
        if tag!='img':
            return
        src=dict(attrs).get('src','')
        if not src.startswith(('data:image/jpeg;base64,','data:image/png;base64,')):
            self.errors.append('image_not_embedded')
            return
        try:
            image=Image.open(io.BytesIO(base64.b64decode(src.split(',',1)[1],validate=True)))
            image.verify()
            self.count+=1
        except Exception as error:
            self.errors.append('invalid_embedded_image:'+str(error))

def validate(T,W):
    name='v7_asset_roster_object_variants_20261012.json'
    paths=[T/'registrations'/name,W/'registrations'/name]
    roster=json.loads(paths[0].read_text())
    summary=json.loads((T/'summary.json').read_text())
    errors=[]
    checks=[]
    def check(value,message):
        if not value:
            errors.append(message)
    check(paths[0].read_bytes()==paths[1].read_bytes(),'roster_copies_differ')
    amendment=json.loads((T/'owner_amendment.json').read_text())
    raw=paths[0].read_text()
    for old in amendment['owner_removed']:
        check(old['asset_id'] not in raw,'owner_removed_still_registered:'+old['asset_id'])
    check(len({r['asset_id'] for r in roster['assets']})==len(roster['assets']),'duplicate_registered_id')
    existing=[r for r in roster['candidates'] if r['origin']=='existing']
    check(len(existing)==30,'A_scope_not_30')
    check(sum(r['silent_decor'] for r in existing)==3,'silent_AC_scope_not_3')
    for row in roster['candidates']:
        if row['object_type']=='air_conditioner':
            check(row['silent_decor'] and row['emitter'] is None,'AC_has_emitter:'+row['asset_id'])
            check(not row['allowed_v7_classes_candidate'],'AC_has_sound_class:'+row['asset_id'])
    shape_counts=collections.defaultdict(list)
    expected_images=sum(len(n.get('screenshots',[])) for r in roster['candidates'] for n in r['native_results'].values() if n)
    for row in roster['assets']:
        aid=row['asset_id']
        m=json.loads(Path(row['asset_metadata']).read_text())
        path=Path(row['derived_glb'])
        check(row['status']=='pass' and not m['preflight_issues'],'admitted_with_failed_preflight:'+aid)
        for repair in row.get('finish_repairs',[]):
            check(Path(repair['receipt']).is_file(),'finish_receipt_missing:'+aid)
        check(path.resolve().is_relative_to(T.resolve()),'derived_outside_owned_root:'+aid)
        size=path.stat().st_size
        vertices,faces=load_glb(path)
        dims=np.ptp(vertices,axis=0)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        check(len(faces)<=100000,'actual_faces_exceed_100k:'+aid)
        check(size<=40_000_000,'actual_glb_exceeds_40MB:'+aid)
        check(size==m['geometry']['glb_bytes'] and len(faces)==m['geometry']['faces'],'mesh_metadata_differs:'+aid)
        check(np.allclose(dims,m['geometry']['dimensions_m'],atol=1e-6),'actual_dimensions_differ:'+aid)
        lo=m['size_basis']['range_min_m'];hi=m['size_basis']['range_max_m']
        check(np.all(dims>=lo) and np.all(dims<=hi),'actual_dimensions_outside_range:'+aid)
        check(m['geometry']['resting_pose']['verdict'] in ['level','acceptable'],'pose_failed:'+aid)
        if row.get('finish_repairs'):
            proof=json.loads(Path(row['finish_repairs'][-1]['receipt']).read_text())
            old_path=Path(proof['source_derived_glb'])
            check(old_path.resolve().is_relative_to(T.resolve()) and old_path.is_file(),'finish_source_missing:'+aid)
            check(hashlib.sha256(old_path.read_bytes()).hexdigest()==proof['source_sha256'],'finish_source_identity_changed:'+aid)
            check(proof['derived_sha256']==digest and Path(proof['derived_glb']).resolve()==path.resolve(),'finish_receipt_derivation_differs:'+aid)
            old_vertices,old_faces=load_glb(old_path)
            check(geometry_match(old_vertices,old_faces,vertices,faces)['status']=='pass','finish_actual_triangle_geometry_changed:'+aid)

        highlights=[];contact=[];labels=[]
        for house in ['mp3d','hm3d']:
            n=row['native_results'].get(house)
            if not n:
                check(False,'native_missing:'+aid+':'+house)
                continue
            check(n['status']=='pass' and n.get('review_method_version')=='object-attachment-body-v5','native_method_or_status_failed:'+aid+':'+house)
            check(n['derived_sha256']==digest and Path(n['derived_glb']).resolve()==path.resolve(),'native_mesh_lineage_differs:'+aid+':'+house)
            deltas=n['placement']['contact_deltas_m']
            contact.extend(abs(v) for v in deltas)
            check(len(deltas)==5 and all(math.isfinite(v) and abs(v)<=.015 for v in deltas),'contact_failed:'+aid+':'+house)
            check(n['buried_3cm_positive_contact'] and not n['moved_6cm_off_support_contact'],'native_contact_controls_failed:'+aid+':'+house)
            shots=n['screenshots']
            check(len(shots)==4 and len({s['name'] for s in shots})==4,'four_native_views_missing:'+aid+':'+house)
            for shot in shots:
                value=shot['production_appearance'].get('highlight_ratio',1)
                highlights.append(value)
                check(0<=value<=.08,'highlights_failed:'+aid+':'+house+':'+shot['name'])
                check(shot['visible_pixels']>700 and shot['border_margin_px']>=3,'visibility_failed:'+aid+':'+house+':'+shot['name'])
                for key in ['rgb_lossless','mask','body_mask']:
                    check(Path(shot[key]).is_file(),'evidence_file_missing:'+aid+':'+key)
            fresh=[]
            thresholds=n.get('color_thresholds',json.loads((T/'color_calibration_refined.json').read_text())['thresholds'])
            for shot in shots:
                rgb=np.asarray(Image.open(shot['rgb_lossless']).convert('RGB'))
                mask=np.asarray(Image.open(shot['body_mask']))>0
                measured=measure_color(rgb,mask,thresholds)
                fresh.append(measured)
                check(measured.get('label')==shot['color'].get('label') and measured.get('status')==shot['color'].get('status'),'RGB_color_measurement_differs:'+aid+':'+house+':'+shot['name'])
            recomputed=summarize_views(fresh,thresholds)
            check(recomputed.get('label')==n['color_label'] and recomputed.get('status')=='pass' and abs(recomputed['main_share']-n['main_color_share'])<1e-9,'RGB_pooled_color_differs:'+aid+':'+house)
            pooled=n['pooled_color']
            check(pooled['status']=='pass' and pooled['main_share']>=.55 and pooled['dominant_agreeing_views']>=2,'body_color_failed:'+aid+':'+house)
            labels.append(n['color_label'])
            if row['silent_decor']:
                check(n['emitter_policy']=='not_applicable_silent_decor' and n['emitter_world_m'] is None,'silent_native_emitter_present:'+aid)
            else:
                check(n['emitter_native_readback_error_m']<=1e-4 and n['emitter_translation_yaw_follow_error_m']<=1e-4,'emitter_follow_failed:'+aid+':'+house)
                point=np.asarray(row['emitter']['local_m'])
                check(np.min(np.linalg.norm(vertices-point,axis=1))<=.06,'emitter_far_from_actual_mesh:'+aid)
            support=n['placement']['support']
            if row['object_type']=='doorbell_chime_unit':
                check(bool(support.get('native_door_anchor')) and support.get('distance_to_native_door_aabb_m',2)<=1.2,'doorbell_door_location_missing:'+aid)
            if row['object_type']=='air_conditioner' and row['form_factor']=='wall_split':
                check(support.get('height_above_floor_m',0)>=2.2,'AC_not_on_high_wall:'+aid)
            if row['object_type']=='air_conditioner' and row['form_factor']=='window_unit':
                check(bool(support.get('native_window_anchor')),'AC_not_at_native_window:'+aid)
        check(labels==[row['label_color']]*2,'house_labels_or_registry_label_differ:'+aid)
        if not row['silent_decor']:
            shape_counts[row['counting_kind']].append(row)
        checks.append({'asset_id':aid,'actual_faces':len(faces),'actual_glb_bytes':size,'actual_dimensions_m':dims.tolist(),'derived_sha256':digest,'contact_max_abs_m':max(contact,default=None),'highlight_max':max(highlights,default=None),'emitter_nearest_vertex_m':None if row['silent_decor'] else float(np.linalg.norm(vertices-np.asarray(row['emitter']['local_m']),axis=1).min())})
    for count in roster['counts']:
        rows=shape_counts[count['object_kind']]
        crossing=collections.defaultdict(collections.Counter)
        for row in rows:
            crossing[row['label_color']][row['form_factor']]+=1
        check(count['usable_appearances']==len(rows),'count_differs:'+count['object_kind'])
        check(count['distinct_forms']==len({r['form_factor'] for r in rows}) and count['distinct_colors']==len(crossing),'form_or_color_count_differs:'+count['object_kind'])
        for cell in count['colors']:
            check(cell['forms']==dict(crossing[cell['color']]),'crossed_cell_differs:'+count['object_kind']+':'+cell['color'])
    check(len(roster['class_coverage'])==13,'sound_class_count_not_13')
    for entry in roster['class_coverage']:
        check(entry['usable_object_appearances']==sum(entry['sound_class'] in r['allowed_v7_classes_candidate'] for r in roster['assets']),'sound_class_count_differs:'+entry['sound_class'])
    for request in roster['generation_requests']:
        stages=request['stage_receipts']
        pixal=stages.get('pixal',{}).get('record',{})
        seeds={x['seed'] for x in pixal.get('attempts',[])}
        check(len(seeds)<=3,'Pixal_more_than_3_seeds:'+request['asset_id'])
        if 'trellis' in stages:
            failures={x['seed'] for x in pixal.get('attempts',[]) if x.get('status')=='fail' and x.get('reason')!='CUDA_OOM'}
            check(len(failures)==3 or pixal.get('setup_failure_elapsed_seconds',0)>=1800,'TRELLIS_without_eligible_failure:'+request['asset_id'])
    audit=ImageAudit();audit.feed((T/'REVIEW.html').read_text())
    errors.extend(audit.errors)
    check(audit.count==expected_images,'HTML_image_count_differs')
    completeness=summary['A_has_30_conclusions'] and summary['B_status']=='pass'
    result={'evidence_integrity_status':'pass' if not errors else 'fail','task_requirements_status':'pass' if completeness and not errors else 'incomplete','errors':errors,'validated_admitted_assets':len(checks),'embedded_images':audit.count,'actual_glb_checks':checks,'sources':[str(p) for p in paths]+[str(T/'summary.json'),str(T/'REVIEW.html')]}
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task-root',type=Path,required=True)
    p.add_argument('--worktree',type=Path,required=True)
    a=p.parse_args()
    result=validate(a.task_root,a.worktree)
    dump(a.task_root/'bundle_validation.json',result)
    progress(a.task_root,'交付证据校验：'+result['evidence_integrity_status']+'；任务标准：'+result['task_requirements_status'],str(a.task_root/'bundle_validation.json'),'修复真实错误或继续填补明确缺口；最终核对测试、提交与作业清理')
    print(json.dumps({k:result[k] for k in ['evidence_integrity_status','task_requirements_status','validated_admitted_assets','embedded_images','errors']},ensure_ascii=False))
    return 1 if result['errors'] else 0
if __name__=='__main__':
    raise SystemExit(main())
