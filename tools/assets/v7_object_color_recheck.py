"""Refine neutral chroma on known-shirt training pixels and reproducibly remeasure native body ROIs."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from v7_object_color import classify,measure,summarize_views
from v7_object_native import body_mask_from_native_depth
from v7_object_prepare import dump
from v7_object_progress import append as progress

def refine(base_path,out):
    base=json.loads(base_path.read_text())
    records=[]
    for row in base['measurements']:
        rgb=np.asarray(Image.open(row['rgb']).convert('RGB'))
        mask=np.asarray(Image.open(row['mask']))>0
        pixels=rgb[mask]
        # Selection uses training pixels only, with equal weight for each known family.
        sampled=pixels[::max(1,len(pixels)//6000)]
        records.append((row,rgb,mask,sampled))
    results=[];best=None
    for chroma in [8,10,12,15,18,22]:
        t=dict(base['thresholds'],neutral_chroma_max=chroma)
        scores={};correct=0;smallest=1.
        for row,rgb,mask,pixels in records:
            if row['split']!='train':
                continue
            labels=classify(pixels,t)
            expected_share=float(np.mean(labels==row['expected']))
            scores.setdefault(row['expected'],[]).append(expected_share)
            correct+=expected_share>=t['main_share_min']
            smallest=min(smallest,expected_share)
        balanced=float(np.mean([np.mean(values) for values in scores.values()]))
        score=(correct,balanced,smallest)
        results.append({'neutral_chroma_max':chroma,'correct_training_samples':correct,'class_balanced_expected_pixel_share':balanced,'minimum_expected_share':smallest})
        if best is None or score>best[0]:
            best=(score,t)
    selected=best[1];rows=[]
    for row,rgb,mask,pixels in records:
        m=measure(rgb,mask,selected)
        rows.append({'rgb':row['rgb'],'mask':row['mask'],'expected':row['expected'],'split':row['split'],**m,'correct':m.get('label')==row['expected'] and m['status']=='pass'})
    held=[r for r in rows if r['split']=='held_out'];accuracy=sum(r['correct'] for r in held)/len(held)
    report={'status':'pass' if accuracy>=.95 else 'fail','reference_root':base['reference_root'],'thresholds':selected,'known_samples':len(rows),'held_out_samples':len(held),'held_out_accuracy':accuracy,'required_held_out_accuracy':.95,'selection':'training full-pixel class-balanced expected-family share; no object pixels or held-out labels used to select thresholds; original brightness boundaries retained','training_grid':results,'prior_calibration':str(base_path),'gray_beige_wood_calibration':base['gray_beige_wood_calibration'],'measurements':rows}
    dump(out,report)
    print(json.dumps({k:report[k] for k in ['status','thresholds','known_samples','held_out_samples','held_out_accuracy']}))
    return report

def recheck_record(raw,meta,T,cal,cal_path):
    if len(raw.get('screenshots',[]))!=4:
        return copy.deepcopy(raw)
    required=['depth','camera_projection','camera_world_from_sensor']
    if not all(all(k in s for k in required) for s in raw['screenshots']):
        raise ValueError('Native v5 depth/camera evidence missing: '+raw['asset_id'])
    identity={'derived_sha256':raw['derived_sha256'],'body_method':('native_surface_normals_v5_closed_lid' if meta['object_type']=='security_keypad' and 'flip_closed' in meta.get('form_factor','') else 'native_surface_normals_v5_final'),'thresholds':cal['thresholds'],'functional_front_axis':meta.get('functional_front_axis','positive-x'),'raw_view_rgb_paths':[s['rgb_lossless'] for s in raw['screenshots']],'placement':raw['placement'],'form_factor':meta['form_factor']}
    D=T/'metrics/native_body_v5_final'/raw['house']/raw['asset_id']
    cached=D/'record.json'
    if cached.exists():
        r=json.loads(cached.read_text())
        if r.get('metric_input_identity')==identity:
            return r
    record=copy.deepcopy(raw);record['metric_input_identity']=identity
    for shot in record['screenshots']:
        rgb=np.asarray(Image.open(shot['rgb_lossless']).convert('RGB'))
        semantic=np.asarray(Image.open(shot['mask']))>0
        body=body_mask_from_native_depth(semantic,np.load(shot['depth']),shot['camera_projection'],shot['camera_world_from_sensor'],record['placement']['root_m'],record['placement']['yaw_rad'],meta)
        D.mkdir(parents=True,exist_ok=True)
        path=D/(shot['name']+'_body.png')
        Image.fromarray(body.astype('uint8')*255).save(path)
        shot['previous_body_mask']=shot['body_mask'];shot['body_mask']=str(path)
        shot['previous_color_measurement']=shot['color'];shot['color']=measure(rgb,body,cal['thresholds'])
    pooled=summarize_views([s['color'] for s in record['screenshots']],cal['thresholds'])
    record['previous_pooled_color']=record['pooled_color'];record['pooled_color']=pooled
    record['color_label']=pooled['label'];record['main_color_share']=pooled['main_share']
    record['reasons']=[reason for reason in record['reasons'] if reason!='no_dominant_body_color']
    if pooled['status']!='pass':
        record['reasons'].append('no_dominant_body_color')
    record['status']='pass' if not record['reasons'] else 'fail'
    record['color_thresholds']=cal['thresholds'];record['color_calibration_path']=str(cal_path)
    record['color_metric_receipt']=str(cached)
    record['body_roi']='native semantic/depth outward normals v5 final; screen/button/grille surfaces excluded only on true front; coloured sides/back retained; jar excluded; 1px erosion'
    dump(cached,record)
    return record

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task-root',type=Path,required=True)
    p.add_argument('--refine',action='store_true')
    a=p.parse_args();T=a.task_root
    cal_path=T/'color_calibration_refined.json'
    if a.refine:
        assert not cal_path.exists(),'Calibration already completed; do not select again'
        cal=refine(T/'color_calibration.json',cal_path)
        progress(T,'仅按人物上衣训练像素细化中性色彩度边界；留出验证 '+str(cal['held_out_accuracy']),str(cal_path),'校准通过后对已保存的原生深度/RGB机器复算；不根据物件结果改阈值')
        if cal['status']!='pass':
            return 1
    if not cal_path.exists():
        return 0
    cal=json.loads(cal_path.read_text())
    assert cal['status']=='pass'
    source=T/'native_existing_body_v5.json'
    if source.exists():
        metas={r['asset_id']:r for r in json.loads((T/'prepared_existing.json').read_text())}
        rows=[]
        raw_rows=json.loads(source.read_text());repair=T/'native_existing_pose_repair.json'
        if repair.exists():raw_rows+=json.loads(repair.read_text())
        for repair in sorted(T.glob('native_existing_finish*.json')):
            batch=json.loads(repair.read_text())
            if isinstance(batch,dict):continue # matching *_receipt.json is metadata, not records
            if not isinstance(batch,list) or not all(isinstance(r,dict) and 'asset_id' in r and 'house' in r for r in batch):
                raise ValueError('Invalid existing native record batch: '+str(repair))
            raw_rows+=batch
        for raw in raw_rows:
            meta=metas[raw['asset_id']];path=Path(meta['derived_path'])
            if raw.get('derived_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest():
                continue
            rows.append(recheck_record(raw,meta,T,cal,cal_path))
        by={(r['asset_id'],r['house']):r for r in rows};rows=[by[k] for k in sorted(by)]
        dump(T/'native_existing_body_v5_remeasured.json',rows)
        print('EXISTING_BODY_RECHECK',len(rows))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
