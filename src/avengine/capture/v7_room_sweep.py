"""Category witnesses using unchanged native production placement predicates.

A passing witness certifies its specific body in its selected room. It is not
an all-asset claim or a complete audiovisual parent admission.
"""
from copy import deepcopy
from pathlib import Path
from collections import Counter
import gc,json,math,os,resource,time,traceback
import numpy as np
from avengine.capture.v7_phase2_native import NativePhase2Video
from avengine.timeline.continuous_pilot import write_json


def witness_scene(room,assets):
    listener=list(room['preferred_listener_m'])
    scene={'scene_id':'geometry_'+room['room_label'],'listener_m':listener,
           'camera_forward':[0.,0.,-1.],'duration_s':30.,'video_fps':15,
           'camera_yaw_knots':[{'time_s':i*3.,'yaw_deg':i*45.}for i in range(9)],
           'sources':{},'events':[],'window_starts_s':[0.,15.],
           'production_scenario':{},'native_placement_candidates':[]}
    for i,a in enumerate(assets,1):
        a.update(endpoint_id='probe'+str(i),native_instance_id=i)
        scene['sources'][a['endpoint_id']]={'actor_id':i,'asset_id':a['asset_id'],
            'trajectory':[{'time_s':0.,'position_m':listener},{'time_s':30.,'position_m':listener}],
            'native_visual_kind':'registered_drawable','silent_decor':a['silent_decor']}
    return scene


def static_categories(house,room,inputs,catalog,categories,directory,save_samples=False):
    inputs=deepcopy(inputs)
    # Finite census proposals only; the complete parent runtime is untouched.
    inputs['visible_placement_max_proposals_per_asset']=192
    inputs['visible_placement_max_contact_attempts_per_asset']=24
    inputs['room_sweep_one_view_witness']=True
    start=time.monotonic();cpu=time.process_time();directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    by_id={a['asset_id']:a for a in catalog};ids=list(dict.fromkeys([x for v in categories.values()for x in v]+['v7_male_adult_14_blue_v1']))
    assets=[deepcopy(by_id[x])for x in ids];scene=witness_scene(room,assets)
    working=deepcopy(house);working['room']=deepcopy(room)
    video=None;results={}
    try:
        video=NativePhase2Video(working,scene,inputs,assets,source_material_cache_directory=Path(inputs['source_material_cache_directory']))
        entries={e['declaration']['asset_id']:e for e in video.actors};all_entries=list(video.actors)
        for e in all_entries:e['object'].translation=video.mn.Vector3(0.,-1000.,0.)
        for category,choices in categories.items():
            # Camera height is a sampling choice, not a contact/frustum gate.
            scene['listener_m'][1]=float(room['measured_floor_y_m'])+(1.65 if category=='ceiling_object'else .95 if category in {'floor_dog','floor_cat','perched_bird'}else 1.25)
            attempts=[]
            for asset_id in choices:
                entry=entries[asset_id];asset=entry['declaration'];video.actors=[entry]
                attempt_dir=directory/category/asset_id;attempt_dir.mkdir(parents=True,exist_ok=False)
                before=time.monotonic();video.material_checks=[];video.color_checks=[]
                try:
                    checks=video.qualify_standing(scene,attempt_dir)
                    # Quadruped paw readbacks retain the task's stricter 12mm
                    # bound; never accept the native 15mm outer band alone.
                    paw_errors=checks[0]['placement']['contact_deltas_m'][-4:]if category in {'floor_dog','floor_cat'}else []
                    if paw_errors and max(abs(v)for v in paw_errors)>.012:raise ValueError('animal_four_paw_contact_exceeds_12mm')
                    maximum=0;measured_views=[];best_rgb=None;best_yaw=None;readbacks=[]
                    for i in range(8):
                        yaw=i*45.;f=[math.sin(math.radians(yaw)),0.,-math.cos(math.radians(yaw))]
                        count=video._native_pair_pixels(scene,[entry],f)[asset_id];maximum=max(maximum,count)
                        if count<100:continue
                        actor=video._apply_entry(entry,scene);readbacks.append(actor)
                        obs=video.sim.get_sensor_observations();rgb=np.asarray(obs['rgb'])[:,:,:3];semantic=np.asarray(obs['semantic'])
                        video._check_source_frame(rgb,semantic,i)
                        measured_views.append({'yaw_deg':yaw,'visible_pixels':count})
                        if best_rgb is None or count>=max(v['visible_pixels']for v in measured_views):best_rgb=rgb.copy();best_yaw=yaw
                        # One complete, measured camera witness is the census goal.
                        # Full parent encoding still checks all 450 native frames.
                        if inputs['room_sweep_one_view_witness']:break
                    measured=[m for m in video.material_checks if m['status']in {'pass','fail'}]
                    failed=[m for m in measured if m['status']=='fail']
                    if maximum<100:raise ValueError('actual_native_visibility_below_100_pixels')
                    if not measured:raise ValueError('source_material_never_measured')
                    if failed:raise ValueError('native_highlight_or_material_failure')
                    sample=None
                    if save_samples and best_rgb is not None:
                        from PIL import Image
                        sample=attempt_dir/'witness.png';Image.fromarray(best_rgb).save(sample)
                    result={'status':'pass','category':category,'asset_id':asset_id,'placement_checks':checks,
                        'animal_paw_errors_m':paw_errors,'maximum_native_visible_pixels':maximum,
                        'observed_views':measured_views,'material_checks':video.material_checks,'color_checks':video.color_checks,
                        'worst_highlight_ratio':max(m.get('highlight_ratio',0.)for m in measured),
                        'attached_emitter_readbacks':readbacks,'sample_image':str(sample)if sample else None,
                        'sample_yaw_deg':best_yaw,'listener_m':list(scene['listener_m']),'wall_seconds':time.monotonic()-before,
                        'scope':'native category witness only; no RIR, audio, QA or sealed parent',
                        'code_revision':'unchanged native placement plus task quadruped 12mm'}
                except Exception as error:
                    rejected=attempt_dir/'attachment_rejection.json';detail=json.loads(rejected.read_text())if rejected.is_file()else {}
                    result={'status':'fail','category':category,'asset_id':asset_id,'error':str(error),
                        'failure_reason_counts':detail.get('rejection_counts',{}),'native_rejection_receipt':str(rejected)if rejected.is_file()else None,
                        'material_checks':video.material_checks,'wall_seconds':time.monotonic()-before,'traceback':traceback.format_exc()}
                finally:
                    entry['object'].translation=video.mn.Vector3(0.,-1000.,0.)
                write_json(attempt_dir/'result.json',result);attempts.append(result)
                if result['status']=='pass':break
            selected=next((r for r in attempts if r['status']=='pass'),None)
            results[category]={'status':'pass'if selected else 'fail','selected':selected,
                'attempts':[str(directory/category/r['asset_id']/'result.json')for r in attempts],
                'failure_reasons':dict(Counter(r.get('error','')for r in attempts if r['status']!='pass'))}
            write_json(directory/('partial_'+category+'.json'),results)
            print(json.dumps({'room_id':room['room_label'],'category':category,'status':results[category]['status'],'wall_seconds':sum(r['wall_seconds']for r in attempts),'reasons':results[category]['failure_reasons']},ensure_ascii=False),flush=True)
            gc.collect()
        scene['listener_m'][1]=float(room['measured_floor_y_m'])+.95
        from avengine.capture.v7_room_seated_witness import seated_witness
        results['chair_seated']=seated_witness(video,scene,room,entries['v7_male_adult_14_blue_v1'],directory/'chair_seated',save_samples)
        write_json(directory/'partial_chair_seated.json',results)
        print(json.dumps({'room_id':room['room_label'],'category':'chair_seated','status':results['chair_seated']['status'],'reasons':results['chair_seated']['failure_reasons']},ensure_ascii=False),flush=True)
        video.actors=all_entries
        result={'status':'static_complete','house':house['house'],'room_id':room['room_label'],
            'categories':results,'chair_seated_status':results['chair_seated']['status'],
            'room_area_m2':room['floor_area_m2'],'native_renderer':video.receipt,
            'wall_seconds':time.monotonic()-start,'process_cpu_seconds':time.process_time()-cpu,
            'sampling_budget':{'support_proposals_per_asset':192,'native_contact_attempts_per_asset':24,'exhaustion_is_not_proof_of_room_impossibility':True,'single_measured_native_view':True},
            'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'formal_parent_count':0,'GPU_jobs':0}
    except Exception as error:
        result={'status':'blocked_native_initialization','house':house['house'],'room_id':room['room_label'],'categories':results,
            'error':str(error),'traceback':traceback.format_exc(),'wall_seconds':time.monotonic()-start,'process_cpu_seconds':time.process_time()-cpu,'formal_parent_count':0}
    finally:
        if video is not None:video.close()
    if inputs.get('witness_evidence_cache_directory'):
        from avengine.capture.v7_witness_storage import share_identical_witness_files
        sharing=share_identical_witness_files(directory,inputs['witness_evidence_cache_directory'])
        write_json(directory/'evidence_sharing_receipt.json',sharing)
        result['evidence_sharing_receipt']=str(directory/'evidence_sharing_receipt.json')
    write_json(directory/'result.json',result);return result
