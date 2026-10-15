"""Animal motion witnesses and row translation for the shared joint solver.

The core owns camera domains, navigation, distance clocks, sound scheduling and
native preparation. This plugin declares radial and later-still obligations,
uses the unchanged motion verifier, and binds original registered recordings.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import math
import tempfile
import numpy as np

FAMILIES = ('QA-06','QA-15','QA-16','QA-17')
REVISION = 'joint-solved-animal-motion-witnesses-v1'


def program_activity(record, profile):
    from avengine.timeline.v7_motion_source_roles import activity
    spans=activity(record);count=int(record['header']['sample_count'])
    if profile:
        count=int(profile['sample_count'])
        if profile.get('tail_muted'):
            from avengine.timeline.v7_tail_muting import read_tail_muted_program
            _,measured=read_tail_muted_program(record,profile)
            spans=measured['activity_intervals_samples']
        else:
            begin=int(profile.get('crop_start_sample',0))
            spans=[[max(int(a),begin)-begin,min(int(b),begin+count)-begin]for a,b in spans
                   if int(b)>begin and int(a)<begin+count]
    return count,spans


def select_sources(catalog, records, semantic, contract, seed, attempt, kind):
    from avengine.timeline.v7_motion_quartet_template import authored_pairs, choose_template_assets
    from avengine.timeline.v7_motion_source_roles import select_motion_role, select_source_program
    from avengine.timeline.v7_visible_sources import production_audio_record_views
    from avengine.timeline.v7_registered_sampling import semantic_record
    from avengine.timeline.v7_environmental_sources import recording_header
    import soundfile as sf
    pairs = authored_pairs(semantic)
    if not pairs:
        raise ValueError('no complete reviewed companion speech pair')
    pair = pairs[(int(seed)+int(attempt)) % len(pairs)]
    humans = choose_template_assets(catalog, pair, seed, attempt)
    source_views=production_audio_record_views(records)
    role = select_motion_role(kind, humans[0], catalog, source_views, contract, seed, attempt)
    animal, companion = role['asset'], humans[1]
    from avengine.construction.joint.routes import SPEED_BANDS
    speed_kind='slide'if kind=='animal_slide'else animal['species_id']
    lo,hi=SPEED_BANDS[speed_kind]
    # Select a legal actual call for the required motion margin, rather than
    # making every RIR stream cover the longest available recording.
    chosen=None
    caps=(1.9,2.2,2.4)if speed_kind=='slide'else(1.5,1.8,2.4)
    for cap in caps:
        try:
            candidate=select_source_program(source_views,set(animal['allowed_classes']),(hi-.02,hi-.01),maximum_program_s=cap)
        except ValueError:
            continue
        record=recording_header(candidate['record']);profile=deepcopy(candidate.get('program'))
        count,spans=program_activity(record,profile)
        span=(spans[-1][1]-spans[0][0])/16000. if spans else 0.
        required=max(lo+.025,.45 if speed_kind in {'cat','slide'}else lo,.65/(.9*span))if span>0 else float('inf')
        if required<=hi-.01:
            chosen=(record,profile,count,spans);break
    if chosen is None:
        record=recording_header(role['record']);profile=deepcopy(role.get('program'))
        count,spans=program_activity(record,profile)
    else:
        record,profile,count,spans=chosen
    if not spans:
        raise ValueError('registered animal program has no measured activity')
    voice = 'F' if companion['visual_gender']=='female' else 'M'
    speech = recording_header(semantic_record(pair[1], voice))
    samples, rate = sf.read(speech['normalized_path'],dtype='float64')
    nz = np.flatnonzero(samples != 0.)
    if rate != 16000 or samples.ndim != 1 or not len(nz):
        raise ValueError('whole companion speech is not a non-silent 16kHz mono recording')
    speech.setdefault('screens',{})['automatic_activity'] = {
        'activity_intervals_samples':[[int(nz[0]),int(nz[-1]+1)]],
        'method':'exact nonzero envelope; original whole speech retained'}
    return {'animal':animal,'companion':companion,'record':record,'program':profile,
        'animal_activity_s':tuple((int(a)/16000.,int(b)/16000.)for a,b in spans),
        'animal_duration_s':count/16000.,'speech':speech,
        'speech_activity_s':((int(nz[0])/16000.,int(nz[-1]+1)/16000.),),
        'speech_duration_s':len(samples)/16000.,'semantic_sound':pair[1],
        'scenario_id':pair[0]['scenario_id'],'kind':kind}


def body_for_asset(asset, species=None):
    from avengine.construction.joint import BodySpec
    kind = species or ('human' if asset['kind']=='human' else asset['species_id'])
    body = BodySpec.for_species(kind)
    dims = asset.get('dimensions_m')
    if isinstance(dims,(list,tuple)) and len(dims)==3 and np.isfinite(dims).all() and min(dims)>0:
        body = BodySpec.for_species(kind,radius_m=max(body.radius_m,min(dims[0],dims[2])/2),
                                   height_m=max(body.height_m,dims[1]))
    return body


def row_from_plan(plan, data, base_row, catalog, records, contract):
    """Translate solved actors and core-scheduled file starts; choose nothing here."""
    from avengine.timeline.v7_production_inputs import bind_animal_motion
    from avengine.timeline.v7_motion_during_conditions import prepare_motion_during_row
    row = deepcopy(base_row)
    mover, companion = data['animal']['asset_id'], data['companion']['asset_id']
    route = deepcopy(plan.metadata['animal_route'])
    starts = plan.metadata['source_file_starts_s']
    movement_offsets = plan.metadata['movement_offsets_s']
    first = starts[mover][0]
    row.update(schema='v7-visible-registered-pilot-phase2-v1',members=['v0_a0'],trial_only=True,
        asset_ids=[mover,companion],silent_asset_ids=[],
        source_classes={mover:[data['record']['sound_class']],companion:['speech_playback']},
        proposed_sound_asset_ids={mover:data['record']['sound_asset_id'],companion:data['speech']['sound_asset_id']},
        semantic_scenario_id=data['scenario_id'],semantic_sound_ids=[data['speech']['sound_asset_id']],
        semantic_human_asset_ids=[companion],semantic_start_offsets_s=[starts[companion][0]],
        motion_source_start_s={mover:first,companion:starts[companion][0]},
        listener_m=list(plan.camera.position_m),camera_yaw_deg=plan.camera.yaw_deg,
        camera_yaw_knots=[{'time_s':t,'yaw_deg':plan.camera.yaw_deg}for t in (0.,30.)],
        window_count=5,acceptance_window_starts_s=[0.,5.,10.,15.,20.],
        motion_quartet_template_revision=REVISION,scene_constructor='motion_quartet_template',
        motion_source_type=data['kind'],acceptance_allow_partial_families=True,
        native_view_roles={mover:{'view':'inside','required_visible_sometime':True},
                           companion:{'view':'outside','minimum_abs_azimuth_deg':65.}},
        phase2_source_event_schedule_s={companion:starts[companion]},joint_solver_only=True,
        automatic_construction={'revision':REVISION,'family':'motion_animal',
                                'solver_entry':'avengine.construction.joint.solve'})
    if data['program'] is not None:
        row['phase2_animal_program_samples']={mover:deepcopy(data['program'])}
        row['phase2_sound_program_schedule_s']={mover:starts[mover]}
    else:
        row['phase2_whole_source_recordings']={mover:starts[mover]}
    request = bind_animal_motion({'action':'walk','speed_m_s':route['speed_m_s'],
        'scene_speed_range_m_s':route['speed_range_m_s'],'motion_quartet_route':route,
        'allow_registered_walk_tts':False,'maximum_native_route_attempts':1,
        'joint_animal_prepared_geometry':plan.actors[mover].metadata['joint_animal_prepared_geometry']},mover,contract)
    row['motion_requests']={mover:request}
    if plan.metadata.get('negative_companion_motion_request'):
        row['motion_requests'][companion]=deepcopy(plan.metadata['negative_companion_motion_request'])
    conditions=[]
    for index, offset in enumerate(movement_offsets):
        branch=route['emission_slots'][index]['branch']
        end_local=plan.metadata['walk_intervals_s'][index][1]-offset-.1
        for family, answer in [('QA-06','moving'),('QA-15',branch),('QA-16',None),('QA-17','yes')]:
            conditions.append({'qa_id':family,'branch':answer,'target_asset_id':mover,
                'speed_m_s':route['speed_m_s'],'speed_range_m_s':route['speed_range_m_s'],
                'minimum_motion_s':1.5,'end_hold_s':10.-end_local,'offsets_s':[offset],
                'reserve_tail_s':.5,'wet_tail_planning_bound_s':None,'minimum_distance_change_m':.5})
    row['motion_during_condition_requests']=conditions
    row=prepare_motion_during_row(row,catalog,list(records)+[data['record'],data['speech']])
    request=row['motion_requests'][mover]
    request['motion_during_condition_plan_for_fit']=deepcopy(row['motion_during_condition_plan'])
    request['intervals_s']=[[0.,30.]]
    for key in ('straight_motion_windows_s','distance_trend','distance_trend_intervals_s'):
        request.pop(key,None)
    from avengine.timeline.v7_demand import demand_snapshot
    feedback=deepcopy(row.get('scene_demand')or demand_snapshot({},next_questions=70))
    row['scene_demand']={**feedback,'focus':'motion_quartet_template','requested_conditions':['motion_quartet'],
        'condition_families':{'motion_quartet':list(FAMILIES)},
        'required_actual_candidate_families':list(FAMILIES),'required_asset_ids':[mover,companion],
        'constructor':REVISION,'static_animal_call_requested':True}
    row['motion_quartet_construction']={'revision':REVISION,'route':route,
        'independent_moving_bouts':len(movement_offsets),'later_static_call':True,
        'events_are_not_automatically_independent_questions':True}
    return row


class MotionAnimalPlugin:
    def __init__(self, data, base_row, catalog, records, contract, house, runtime, output_dir, *, geometry_preparer=None):
        self.data,self.base_row,self.catalog,self.records,self.contract=data,base_row,catalog,records,contract
        self.house,self.runtime,self.output_dir=house,runtime,Path(output_dir)
        self.geometry_preparer=geometry_preparer
        self.rejections={}
        self.geometry_preparations=0
        self.geometry_cache={}
        self.geometry_failure_cache=set()

    def candidates(self, context, request):
        from avengine.construction.joint import CameraPose,ScenePlan
        from avengine.construction.joint.context import camera_legal
        from avengine.construction.joint.geometry import framing
        from avengine.construction.joint.routes import construct_route,static_actor,SPEED_BANDS,ConstructionError
        from avengine.construction.joint.scheduling import SoundRequirement,schedule_sound_events
        from avengine.construction.joint.certification import certify_plan
        from avengine.assets.animal_motion.joint_execution import prepare_route_geometry
        from avengine.assets.roster_motion.straight_route import _radial_flags
        from shapely.geometry import LineString,Point
        d=self.data;mover=d['animal']['asset_id'];companion=d['companion']['asset_id']
        species='slide'if d['kind']=='animal_slide'else d['animal']['species_id']
        body=body_for_asset(d['animal'],species);human=body_for_asset(d['companion'])
        lo,hi=SPEED_BANDS[species];duration=math.ceil(d['animal_duration_s']*10-1e-9)/10
        activity_span=d['animal_activity_s'][-1][1]-d['animal_activity_s'][0][0]
        speed=max(lo+.025,.45 if species in {'cat','slide'}else lo,.65/(.9*activity_span))
        if speed>hi-.01:return
        start,end=1.3,1.3+duration
        tail=end+.5+1.8;w0,w1=start-.5,tail+.5
        # A crop starting at 5s must not contain only the final low-speed ramp.
        # This extends quiet locomotion, never the recording or the speed band.
        w1=max(w1,5.8);tail=w1-.5
        length=speed*(w1-w0-.3)
        if w1>9.4:return
        bouts=int(request.options.get('movement_bouts',1))
        if bouts not in (1,2):raise ValueError('animal joint planning supports one or two movement bouts')
        if bouts==2:
            from avengine.construction.joint.routes import repeat_bouts
        offsets=[0.]if bouts==1 else[0.,10.]
        points=context.points_m[context.walkable_map.contains_many(context.points_m[:,[0,2]],species)]
        people=context.points_m[context.walkable_map.contains_many(context.points_m[:,[0,2]],'human')]
        seen_cameras=set();emitted=0
        for original_camera in context.cameras:
            origin=np.asarray(original_camera.position_m);camera_key=tuple(origin)
            if camera_key in seen_cameras:continue
            seen_cameras.add(camera_key)
            radii=np.linalg.norm(points[:,[0,2]]-origin[[0,2]],axis=1)
            for index in np.argsort(radii,kind='stable'):
                if not 2.6<=radii[index]<=4.5:continue
                near=points[index].copy();delta=near[[0,2]]-origin[[0,2]];direction=delta/np.linalg.norm(delta)
                far=near.copy();far[[0,2]]+=length*direction
                # Radial family seed: the shared species layer covers the entire
                # camera-to-emitter corridor, including body-end reserve.
                reserve=.15
                extended=far[[0,2]]+reserve*direction
                if not context.walkable_map.segment_free(origin[[0,2]],extended,species):continue
                camera=CameraPose(tuple(origin),math.degrees(math.atan2(direction[0],-direction[1])),width=1280,height=720)
                if not camera_legal(context.room,camera)[0]or any(framing(camera,p,body)[0]!='inside'for p in (near,far)):continue
                line=LineString([near[[0,2]],far[[0,2]]])
                candidates=[]
                for p in people:
                    bearing=math.degrees(math.atan2(float((p-origin)@camera.right),float((p-origin)@camera.forward)))
                    if abs(bearing)<68.:continue  # Native outside role requires >=65deg; retain emitter-offset margin.
                    if (np.linalg.norm(p[[0,2]]-origin[[0,2]])<1.6 or line.distance(Point(*p[[0,2]]))<1.1
                        or framing(camera,p,human)[0]!='outside'):continue
                    candidates.append(p)
                if not candidates:continue
                if context.raycaster is not None:
                    # Core stage rays are only a cheap sound-placement domain.
                    # The oracle still uses actual native emitters and original RIRs.
                    mouth=np.asarray(d['companion'].get('mouth_local_glb_m',[0.,human.height_m*.8,0.]),float)
                    delta=np.asarray(candidates)+mouth-origin
                    hit,_=context.raycaster.first_hit(origin,delta)
                    clear=hit>=np.linalg.norm(delta,axis=1)-.04
                    candidates=[point for point,passed in zip(candidates,clear)if passed]
                if not candidates:continue
                candidates.sort(key=lambda p:(-line.distance(Point(*p[[0,2]])),float(p[0]),float(p[2])))
                for standing in candidates[:4]:
                    waypoints=[far,near]if request.branch=='nearer'else[near,far]
                    try:
                        actor=construct_route(context,waypoints,species=species,speed_m_s=speed,start_s=w0,
                                              camera=camera,entity_id=mover,body=body)
                        if actor.metadata['validation']['max_snap_horizontal_m']>.002:continue
                        if bouts==2:actor=repeat_bouts(actor,offsets_s=(0.,10.),context=context,camera=camera)
                        speaker=static_actor(companion,standing,human)
                        if request.options.get('require_static_negative_comparator'):
                            from avengine.assets.animal_motion.negative_witness import construct_negative_companion
                            speaker=construct_negative_companion(context,camera,standing,human,actor,
                                entity_id=companion,animal_call_s=21.3,animal_duration_s=duration,admitted_asset=d['companion'])
                        preliminary=ScenePlan(camera,{mover:actor,companion:speaker})
                        if not certify_plan(context,preliminary).passed:continue
                    except ConstructionError as error:
                        if request.options.get('require_static_negative_comparator'):
                            self.rejections[str(error)]=self.rejections.get(str(error),0)+1
                        continue
                    target_starts=[offset+1.3 for offset in offsets]+[21.3]
                    silence=((4.2,6.6),(14.2,16.6),(24.2,26.6))
                    requirements=[]
                    for i,t in enumerate(target_starts):
                        allowed=(t-.2,t+duration+.3)
                        requirements.append(SoundRequirement(mover,d['record']['sound_asset_id'],d['record']['sound_class'],
                            d['animal_activity_s'],(t,t+.1),(allowed,),exclusive_group='scene_voices'))
                    for offset in (0.,10.,20.):
                        latest=min(7.2,9.45-d['speech_duration_s'])
                        if latest<7.:break
                        requirements.append(SoundRequirement(companion,d['speech']['sound_asset_id'],'speech_playback',
                            d['speech_activity_s'],(offset+7.,offset+latest),((offset+7.,offset+9.6),),exclusive_group='scene_voices'))
                    else:
                        schedule=schedule_sound_events(requirements,silent_intervals_s=silence,start_grid_hz=10,minimum_gap_s=.1)
                        if schedule.status!='solved':continue
                        selected=list(schedule.starts_s);file_starts={mover:selected[:len(target_starts)],companion:selected[len(target_starts):]}
                        xz=actor.positions_m[:,[0,2]];slots=[]
                        for j,offset in enumerate(offsets):
                            sound_start=file_starts[mover][j];sound_end=sound_start+duration
                            q=offset+tail;k,e,query=(round(value*30)for value in(sound_start,sound_end,q))
                            flags=_radial_flags(xz,origin[[0,2]],k,e,query)
                            # The original verifier below reads the exact measured
                            # activity window; no second middle-60-percent gate.
                            slots.append({'start_s':sound_start,'end_s':sound_end,'post_until_s':q,'offset_s':offset,
                                'branch':request.branch if j%2==0 else('farther'if request.branch=='nearer'else'nearer'),
                                'radial_change_m':flags[2],'radial_cos':flags[3],'qa06':True,'qa15':True,'qa16':True,'qa17':True,
                                'leg':[round((offset+w0)*30),round((offset+w1)*30)],
                                'window_frames':[round((offset+w0)*30),round((offset+w1)*30)],'constructed_families':'all4'})
                        else:
                            window={'walk_start_s':w0,'walk_end_s':w1,'ramp_s':.3,'speed_m_s':speed,'length_m':length,
                                'emission_s':[slots[0]['start_s'],slots[0]['end_s']],'silent_query_until_s':tail,
                                'heading_rad':float(actor.headings_rad[0]),'straight_line':True,'shape':None}
                            route={'reflect_walk':True,'joint_exact_route':True,'positions_xz_m':xz.tolist(),
                                'travel_yaws_rad':actor.headings_rad.tolist(),'listener_m':list(camera.position_m),'camera_yaw_deg':camera.yaw_deg,
                                'speed_m_s':speed,'speed_range_m_s':[lo,hi],'species':species,'template':'joint',
                                'emission_slot':slots[0],'emission_slots':slots,'short_window_walk':window,
                                'walkable_map_path':str(context.walkable_map.path),'walkable_map_room_id':context.walkable_map.metadata['room_id'],
                                'walkable_map_version':context.walkable_map.metadata['version'],'walkable_map_bypassed':False,
                                'candidate_count':emitted+1,'candidate_trials':[],'selected_attempt':'joint-animal-'+str(emitted),
                                'animal_native_preflight_required':True,'require_rest_poses_in_frame':True,
                                'animal_stationary_turn_intervals_s':actor.metadata.get('stationary_turn_intervals_s',[])}
                            # The original animal pose preparation excludes unplaced
                            # companions. Reuse it only for the identical animal clock,
                            # body/source and observer; final native scene physics is rerun.
                            import hashlib
                            key=(mover,repr(camera),hashlib.sha256(
                                np.asarray(actor.positions_m,dtype='<f8').tobytes()+
                                np.asarray(actor.headings_rad,dtype='<f8').tobytes()).hexdigest())
                            if key in self.geometry_failure_cache:continue
                            cached=self.geometry_cache.get(key)
                            if cached is not None:
                                actor.positions_m=cached[0].copy()
                                actor.metadata.update(cached[1])
                            else:
                                self.output_dir.mkdir(parents=True,exist_ok=True)
                                directory=Path(tempfile.mkdtemp(prefix='geometry_',dir=self.output_dir));directory.rmdir()
                                if self.geometry_preparations>=int(self.runtime.get('joint_max_geometry_preparations',4)):return
                                self.geometry_preparations+=1
                                preparer=self.geometry_preparer or prepare_route_geometry
                                try:preparer(context,self.house,self.runtime,d['animal'],self.contract['animal_motion'][mover],actor,route,directory)
                                except ValueError as error:
                                    self.geometry_failure_cache.add(key)
                                    self.rejections[str(error)]=self.rejections.get(str(error),0)+1;continue
                                self.geometry_cache[key]=(actor.positions_m.copy(),{
                                    name:actor.metadata[name]for name in ('native_yaw_rad','joint_animal_prepared_geometry')})
                            plan=ScenePlan(camera,{mover:actor,companion:speaker},schedule.events,metadata={
                                'animal_route':route,'source_file_starts_s':file_starts,'movement_offsets_s':offsets,
                                'walk_intervals_s':actor.metadata['walk_intervals_s'],'roles':{mover:'inside',companion:'outside'},
                                'third_static_call_s':file_starts[mover][-1],'sound_schedule':schedule.diagnostics})
                            if request.options.get('require_static_negative_comparator'):
                                from avengine.assets.animal_motion.negative_witness import companion_motion_request
                                plan.metadata['negative_companion_motion_request']=companion_motion_request(
                                    speaker,camera,context.walkable_map,file_starts[companion][-1],d['speech_duration_s'])
                            emitted+=1
                            yield plan
                            if emitted>=int(request.options.get('candidate_limit',24)):return

    def verify(self, context, request, plan):
        from avengine.construction.joint import CheckResult
        from avengine.timeline.v7_motion_during_conditions import verify_planned_trajectories
        from avengine.assets.animal_motion.route_solver import delivery_speed_reports
        d=self.data;mover=d['animal']['asset_id']
        row=row_from_plan(plan,d,self.base_row,self.catalog,self.records,self.contract)
        tracks={aid:np.asarray(actor.positions_m,float).copy()for aid,actor in plan.actors.items()}
        mouth=float(d['animal'].get('mouth_local_glb_m',d['animal'].get('emitter_local_m',[0.,.3,0.]))[1])
        tracks[mover][:,1]=np.mean(tracks[mover][:,1])+mouth
        family_all={family:True for family in FAMILIES}
        for index,item in enumerate(row['motion_during_condition_plan']['plans']):
            verdict=verify_planned_trajectories({'plans':[item]},tracks,list(plan.camera.position_m))
            for family,ok in verdict.items():
                family_all[family] = family_all[family] and bool(ok)
                yield CheckResult('animal:event:%d:%s'%(index,family),'pass'if ok else'fail',
                                  'original_motion_verifier',evidence={'offsets_s':item['offsets_s']})
        # Certify the actual admitted muzzle stream with the catalog's own
        # distance predicate as well; root-only planning is insufficient.
        import json
        from avengine.qa.unified_catalog import distance_trend_during_window
        prepared=json.loads(Path(plan.actors[mover].metadata['joint_animal_prepared_geometry']).read_text())['record']
        local=np.asarray([frame['mouth_m']for frame in prepared['source_frames']],float)
        roots_full=np.asarray(prepared['world_roots_m'],float)
        yaws=np.asarray(prepared.get('yaw_rad_per_frame',[prepared['yaw_rad']]*901),float)
        cosine,sine=np.cos(yaws),np.sin(yaws)
        mouth_world=np.column_stack((cosine*local[:,0]+sine*local[:,2],local[:,1],-sine*local[:,0]+cosine*local[:,2]))+roots_full
        sampled=mouth_world[::2]
        listener=np.asarray(plan.camera.position_m,float)
        facts={'time':{'frame_count':len(sampled),'frame_rate_hz':15.},
            'actors':{mover:{'emitter_positions_m':sampled.tolist()}},
            'listener':{'status':'pass','positions_m':np.tile(listener,(len(sampled),1)).tolist()}}
        radial=np.linalg.norm(mouth_world-listener,axis=1);clock=np.arange(901)/30.
        for index,slot in enumerate(plan.metadata['animal_route']['emission_slots']):
            start=slot['start_s']+d['animal_activity_s'][0][0]
            stop=slot['start_s']+d['animal_activity_s'][-1][1]
            measured=distance_trend_during_window(facts,mover,[math.floor(start*15),math.ceil(stop*15)],min_net_change_m=.5)
            ok=measured['verdict']==slot['branch']
            family_all['QA-15']=family_all['QA-15'] and ok
            slot['radial_change_m']=measured.get('endpoint_delta_m',slot['radial_change_m'])
            slot['prepared_muzzle_distance_trend']=measured
            yield CheckResult('animal:muzzle_distance:%d'%index,'pass'if ok else'fail',
                'unchanged_catalog_distance_trend_during_window',evidence=measured)
            anchor_time=slot['start_s']+d['animal_duration_s']
            anchor=float(np.interp(anchor_time,clock,radial))
            upper=min(plan.metadata['walk_intervals_s'][index][1]-.3,
                plan.metadata['source_file_starts_s'][d['companion']['asset_id']][index]-.2)
            queries=[]
            for second in range(math.ceil(anchor_time+.6),math.floor(upper)+1):
                delta=float(np.interp(second,clock,radial))-anchor
                if abs(delta)>=.3 and (delta<0)==(slot['branch']=='nearer'):
                    queries.append({'time_s':second,'delta_m':delta})
            family_all['QA-16']=family_all['QA-16'] and bool(queries)
            yield CheckResult('animal:integer_muzzle_query:%d'%index,'pass'if queries else'fail',
                'actual_prepared_muzzle_clock_whole_second_margin',evidence={'queries':queries,'margin_m':.3})
        route=plan.metadata['animal_route'];speed=delivery_speed_reports(route,route['species'])
        yield CheckResult('animal:all_delivery_speeds','pass'if all(v['status']=='pass'for v in speed)else'fail',
                          'unchanged_export_speed_ruler',evidence={'windows':speed})
        roots=plan.actors[mover].positions_m;begin=round(plan.metadata['third_static_call_s']*30)
        held=bool(np.max(np.abs(roots[begin:]-roots[begin]))<1e-8)
        yield CheckResult('animal:later_still_call','pass'if held else'fail',
                          'zero_body_translation_during_and_after_registered_call',
                          evidence={'QA-06':'still','QA-17':'no','start_s':plan.metadata['third_static_call_s']})
        if request.options.get('require_static_negative_comparator'):
            from avengine.assets.animal_motion.negative_witness import verify_negative_comparator
            yield from verify_negative_comparator(plan,mover,d['companion']['asset_id'],
                plan.metadata['third_static_call_s'],d['animal_duration_s'])
        row['motion_quartet_planned_verification']=family_all
        row['motion_requests'][mover]['motion_quartet_route']=deepcopy(plan.metadata['animal_route'])
        row['motion_quartet_construction']['route']=deepcopy(plan.metadata['animal_route'])
        row['joint_planned_sound_events']=[asdict(event)for event in plan.sounds]
        plan.metadata['row']=row


def construct_motion_animal_scene(room, house, state, catalog, records, semantic, contract, runtime,
                                  *, scene_id, seed, family='QA-06', attempt=0,
                                  prepare_and_verify=None):
    """The sole formal constructor: solve, native oracle, then the same payload."""
    from functools import partial
    from avengine.construction.joint import FamilyRequest,solve
    from avengine.construction.joint.context import build_room_context
    from avengine.timeline.v7_production_inputs import planning_catalog
    from avengine.assets.animal_motion.native_preflight import animal_native_checks
    kind=runtime.get('motion_animal_source_type',('dog_walk','cat_walk','animal_slide')[int(seed)%3])
    if kind not in {'dog_walk','cat_walk','animal_slide'}:raise ValueError('unsupported actual animal motion source type')
    room_id=room.get('room_label',room.get('room_id'))
    room=deepcopy(next(item for item in house['checked_rooms']if item['room_label']==room_id))
    if house['house'] in set(contract.get('test_houses',[])):
        raise ValueError('test houses are excluded from animal production')
    base={'schema':'v7-visible-registered-pilot-phase2-v1','split':'train','house':house['house'],
          'room_id':room.get('room_label',room.get('room_id')),'scene_id':scene_id,'scene_seed':int(seed),
          'question_quota_base':10,'questions_per_window':14,'window_count':5,'motion_source_type':kind}
    if house.get('split')!='train':raise ValueError('animal production accepts only the authorized training houses')
    from avengine.timeline.v7_demand import demand_snapshot
    base['scene_demand']=demand_snapshot(state,next_questions=base['window_count']*base['questions_per_window'])
    catalog=planning_catalog(base,catalog,contract)
    preferred=runtime.get('motion_animal_preferred_asset_ids')
    if preferred is not None:
        if not isinstance(preferred,list) or not preferred or any(not isinstance(key,str)for key in preferred):
            raise ValueError('animal preferred asset inputs require explicit nonempty registered IDs')
        if not any(a['kind']=='animal'and a['asset_id']in preferred for a in catalog):
            raise ValueError('requested original animal body is unavailable in this legal room catalog')
        catalog=[a for a in catalog if a['kind']!='animal'or a['asset_id']in preferred]
    data=select_sources(catalog,records,semantic,contract,seed,attempt,kind)
    output=runtime.get('joint_planning_output_root')
    if not output:raise ValueError('animal joint construction requires an explicit caller-owned planning output root')
    output=Path(output)/scene_id;output.mkdir(parents=True,exist_ok=True)
    branch=runtime.get('motion_animal_radial_branch','nearer'if (int(seed)+int(attempt))%2==0 else'farther')
    if branch not in {'nearer','farther'}:raise ValueError('animal radial answer must be fixed before solve')
    request=FamilyRequest('motion_animal',branch,(data['animal']['asset_id'],),{
        'require_native_preflight':True,'movement_bouts':int(runtime.get('motion_animal_movement_bouts',1)),
        'candidate_limit':int(runtime.get('joint_candidate_limit',24)),
        'families':FAMILIES,'later_static_call':True,
        'require_static_negative_comparator':bool(runtime.get('motion_animal_static_negative_comparator',False))})
    plugin=MotionAnimalPlugin(data,base,catalog,records,contract,house,runtime,output/'geometry')
    with build_room_context(room,house=house,runtime=runtime,cache_dir=output/'room_cache',
            camera_limit=36,grid_step_m=.25,camera_heights_m=(1.1,),width=1280,height=720)as context:
        if prepare_and_verify is None:
            from avengine.construction.joint.native import prepare_native
            def oracle(ctx,req,plan):
                from avengine.timeline.v7_production_inputs import attach_runtime_contract
                row_factory=lambda selected:attach_runtime_contract(deepcopy(selected.metadata['row']),runtime)
                family_check=lambda video,scene,entries:animal_native_checks(video,scene,entries,
                    plan=plan,row=plan.metadata['row'])
                return prepare_native(ctx,req,plan,row_factory=row_factory,house=house,catalog=catalog,
                    records=records,semantic=semantic,runtime=runtime,output_dir=output/'native_preparations',
                    family_check=family_check,roles=plan.metadata['roles'])
        else:
            oracle=prepare_and_verify
        result=solve(context,request,plugin,budget_s=float(runtime.get('joint_budget_s',900.)),
            max_candidates=int(runtime.get('joint_max_candidates',24)),prepare_and_verify=oracle,
            max_native_preparations=int(runtime.get('joint_max_native_preparations',3)))
    report={'status':result.status,'reasons':result.reasons,'diagnostics':result.diagnostics,
            'certificate':asdict(result.certificate)if result.certificate else None,
            'source_type':kind,'requested_branch':branch,'family':family,
            'formal_native_prepared':result.prepared is not None,
            'geometry_preparations':plugin.geometry_preparations,'geometry_conflicts':plugin.rejections,
            'source_asset_id':data['animal']['asset_id'],'sound_asset_id':data['record']['sound_asset_id'],
            'actual_source_activity_s':data['animal_activity_s'],'actual_source_program_duration_s':data['animal_duration_s'],
            'preferred_original_asset_ids':preferred}
    (output/'joint_solve_result.json').write_text(__import__('json').dumps(report,indent=2)+'\n')
    if result.status!='certified' or result.prepared is None:
        raise ValueError('animal joint solver did not prepare a certified scene: '+str(result.reasons))
    prepared=result.prepared.payload
    row=prepared.row
    row['joint_solver_certificate']=report
    row['joint_prepared_scene_required']=True
    return {'profile':'phase2','row':row,'prepared':prepared,'prepared_native':prepared,
            'joint_plan':result.plan,'joint_certificate':result.certificate}
