"""Complete registered speech and original four-family witnesses for endpoint plans."""
from __future__ import annotations
from dataclasses import asdict
from copy import deepcopy
from collections import defaultdict,Counter
import math
import numpy as np

FAMILIES=("QA-06","QA-15","QA-16","QA-17")
REVISION="human-core-endpoints-full-program-v1"


def complete_pairs(manifest):
    """All distinct registered voices/roles; no private utterance-length band."""
    grouped=defaultdict(list)
    for sound in manifest["sounds"]:
        grouped[sound["scenario_id"]].append(sound)
    pairs=[]
    for values in grouped.values():
        for a in values:
            for b in values:
                if (a["role_id"]!=b["role_id"] and
                    a["voice_reference"]["speaker_id"]!=b["voice_reference"]["speaker_id"]):
                    pairs.append((a,b))
    return sorted(pairs,key=lambda p:(sum(float(s["synthesis_readback"]["duration_s"])for s in p),
                                      tuple(s["sound_asset_id"]for s in p)))


def human_roles(catalog,pair,seed):
    """Real registered bodies, independent choices; no extra people or color gate."""
    rng=np.random.default_rng(seed);chosen=[]
    for sound in pair:
        pool=[a for a in catalog if a.get("kind")=="human"
              and a.get("walk_allowed") and a.get("visual_gender")==sound["voice_gender"]
              and a["asset_id"]not in {x["asset_id"]for x in chosen}
              and "speech_playback"in a.get("allowed_classes",["speech_playback"])]
        pool.sort(key=lambda a:a["asset_id"])
        if not pool:raise ValueError("no registered human for the actual speech/action binding")
        chosen.append(pool[int(rng.integers(len(pool)))])
    return chosen


def measured_record(sound,asset):
    import soundfile as sf
    from avengine.timeline.v7_registered_sampling import semantic_record
    record=semantic_record(sound,"F"if asset["visual_gender"]=="female"else"M")
    samples,rate=sf.read(record["normalized_path"],dtype="float64")
    nonzero=np.flatnonzero(samples!=0.)
    if rate!=16000 or samples.ndim!=1 or not len(nonzero):
        raise ValueError("registered complete speech has no measured mono 16kHz activity")
    record["header"]={"sample_rate_hz":rate,"sample_count":len(samples),"channels":1}
    record.setdefault("screens",{})["automatic_activity"]={
        "activity_intervals_samples":[[int(nonzero[0]),int(nonzero[-1]+1)]],
        "method":"complete admitted waveform exact nonzero envelope"}
    return record


def emitter_tracks(actors,assets):
    result={}
    byid={a["asset_id"]:a for a in assets}
    from avengine.capture.v7_visible_attachment import rotation
    for aid,actor in actors.items():
        asset=byid[aid]
        local=np.asarray(asset.get("emitter_local_m",asset.get("mouth_local_glb_m")),float)
        if local.shape!=(3,)or not np.isfinite(local).all():
            raise ValueError("registered human lacks its actual native mouth landmark")
        headings=np.asarray(actor.headings_rad if actor.headings_rad is not None else
                            np.full(len(actor.positions_m),-float(actor.metadata["native_yaw_rad"])-math.pi/2))
        offsets=np.asarray([rotation(-float(h)-math.pi/2)@local for h in headings])
        result[aid]=np.asarray(actor.positions_m)+offsets
    return result


def speech_slots(actor,camera,record,branch,*,crop_offset=0.,companion_duration_s):
    """Rank native 100ms clock slots by original-answer geometric margins."""
    from avengine.timeline.v7_motion_during_conditions import smooth_track
    from avengine.assets.roster_motion.distance_constraints import matches_radial_window
    from avengine.qa.unified_catalog import POST_SOUND_DISTANCE_MARGIN_M
    header=record["header"];duration=math.ceil(header["sample_count"]/1600.)/10.
    activity=record["screens"]["automatic_activity"]["activity_intervals_samples"]
    begin=activity[0][0]/16000.;end=activity[-1][1]/16000.
    reply=math.floor((10.-companion_duration_s-.5-1/15.)*10.+1e-9)/10.
    positions=np.asarray(actor.positions_m,float)
    # The real mouth landmark is added by the caller before final verification.
    track=positions[round(crop_offset*30):round((crop_offset+10)*30):2]
    if len(track)!=150:return []
    smooth=smooth_track(track)
    radii=np.linalg.norm(smooth-np.asarray(camera.position_m),axis=1)
    raw_radii=np.linalg.norm(track-np.asarray(camera.position_m),axis=1)
    movement=np.linalg.norm(np.diff(track[:,[0,2]],axis=0),axis=1)>1e-8
    sign=-1 if branch=="nearer"else 1
    options=[]
    for tick in range(max(0,math.floor((10.-duration)*10.))+1):
        start=tick/10.;lo=max(0,math.ceil((start+begin)*15.));hi=min(149,math.floor((start+end)*15.))
        if hi<=lo or not movement[max(0,lo-1):hi].any():continue
        values=radii[lo:hi+1]
        during=sign*float(values[-1]-values[0])
        during_pass=matches_radial_window(values,sign,.5)
        anchor=float(np.interp(start+end,np.arange(150)/15.,raw_radii))
        queries=[]
        for second in range(math.ceil(start+end+.5),min(9,math.floor(reply))+1):
            frame=second*15
            if frame>=len(track):continue
            delta=sign*float(raw_radii[frame]-anchor)
            moving=bool(movement[min(frame,len(movement)-1)])
            if delta>=POST_SOUND_DISTANCE_MARGIN_M and moving:
                queries.append((second,delta))
        # The original compiler requires two displayable integer query marks.
        pairs=[(a,b)for a,b in zip(queries,queries[1:])if b[0]==a[0]+1]
        if pairs:
            first,second=pairs[0]
            queries=[q for q in queries if q[0]>=first[0]]
        has_query_window=bool(pairs)
        rank=(int(during_pass)+2*int(has_query_window),min(during-.5,max((v-POST_SOUND_DISTANCE_MARGIN_M for _,v in queries),default=-math.inf)),during)
        options.append({"start_s":crop_offset+start,"local_start_s":start,"reply_local_s":reply,
                        "duration_s":duration,"queries":queries,"rank":rank,
                        "predicted_families":{"QA-06":True,"QA-15":bool(during_pass),
                                              "QA-16":has_query_window,"QA-17":has_query_window}})
    return sorted(options,key=lambda x:x["rank"],reverse=True)


def base_row(room,assets,pair,scene_id,seed,camera,slots):
    ids=[a["asset_id"]for a in assets];a,b=ids
    da=math.ceil(float(pair[0]["synthesis_readback"]["duration_s"])*10.)/10.
    db=math.ceil(float(pair[1]["synthesis_readback"]["duration_s"])*10.)/10.
    starts=[slots[a]["start_s"],slots[a]["reply_local_s"]]
    middle=next(iter(slots[a].get("additional_events",[])),None)
    middle_start=middle["start_s"]if middle else 10.+slots[a]["local_start_s"]
    schedule={a:[starts[0],middle_start,20.+slots[b]["reply_local_s"]],
              b:[slots[a]["reply_local_s"],10.+slots[a]["reply_local_s"],slots[b]["start_s"]]}
    # The second actor's companion reply is the first actor's complete waveform.
    row={"schema":"v7-visible-registered-pilot-phase2-v1","split":"train",
         "house":room["house"],"room_id":room["room_id"],"scene_seed":int(seed),"scene_id":scene_id,
         "independent_parent_scene_id":scene_id,"asset_ids":ids,"silent_asset_ids":[],
         "source_classes":{aid:["speech_playback"]for aid in ids},
         "semantic_scenario_id":pair[0]["scenario_id"],
         "semantic_sound_ids":[s["sound_asset_id"]for s in pair],
         "semantic_human_asset_ids":ids,"semantic_start_offsets_s":starts,
         "proposed_sound_asset_ids":{aid:s["sound_asset_id"]for aid,s in zip(ids,pair)},
         "motion_source_start_s":dict(zip(ids,starts)),
         "phase2_authored_speech_schedule_s":schedule,"phase2_source_event_schedule_s":schedule,
         "listener_m":list(camera.position_m),"camera_yaw_deg":float(camera.yaw_deg),
         "camera_yaw_knots":[{"time_s":t,"yaw_deg":float(camera.yaw_deg)}for t in (0.,30.)],
         "motion_source_type":"human_speech","scene_constructor":"human_endpoint_joint",
         "motion_quartet_template_revision":REVISION,
         "members":["v0_a0"],"trial_only":True,"window_count":5,"questions_per_window":14,
         "question_quota_base":0,"acceptance_window_starts_s":[0.,5.,10.,15.,20.],
         "acceptance_allow_partial_families":True,
         "automatic_construction":{"revision":REVISION,"family":"motion_human",
              "constructor":"core_endpoint_roles","required_role_count":len(ids)},
         "primary_motion_speech_anchors_s":{aid:slots[aid]["start_s"]for aid in ids},
         "motion_requests":{},"scene_demand":{},
         "human_motion_event_count":2,"human_motion_strategy":"independent_endpoint_walkers"}
    return row


def attach_routes(row,plan,assets):
    from avengine.assets.roster_motion.navmesh_execution import REVISION as EXECUTION_REVISION
    from avengine.timeline.v7_motion_during_conditions import prepare_motion_during_row
    from avengine.assets.roster_motion.navmesh_execution import prepare_scene_motion
    for aid,actor in plan.actors.items():
        if not np.any(np.linalg.norm(np.diff(actor.positions_m[:,[0,2]],axis=0),axis=1)>1e-8):
            continue
        positions=np.asarray(actor.positions_m);steps=np.linalg.norm(np.diff(positions[:,[0,2]],axis=0),axis=1)
        distance=np.r_[0.,np.cumsum(steps)]
        solved={"positions_m":positions.tolist(),"headings_rad":np.asarray(actor.headings_rad).tolist(),
                "planned_distance_m":distance.tolist(),"camera":asdict(plan.camera)}
        row["motion_requests"][aid]={"action":"walk","allow_registered_walk_tts":True,"speed_m_s":float(actor.metadata["speed_m_s"]),
             "scene_speed_range_m_s":[.8,1.4],"minimum_listener_horizontal_distance_m":1.5,
             "intervals_s":actor.metadata["walk_intervals_s"],"joint_endpoint_route":solved,
             "motion_quartet_route":{"joint_endpoint_clock":True,"positions_xz_m":positions[:,[0,2]].tolist(),
                                     "travel_yaws_rad":np.asarray(actor.headings_rad).tolist()},
             "motion_execution_revision":EXECUTION_REVISION}
        actor.metadata["native_yaw_rad"]=-float(actor.headings_rad[0])-math.pi/2
        actor.metadata["native_yaw_rad_per_frame"]=(-np.asarray(actor.headings_rad)-math.pi/2).tolist()
    row["motion_execution_revision"]=EXECUTION_REVISION
    return prepare_scene_motion(row,assets)


def verify_actor_families(row,plan,assets,records,target,branch,slot,*,funnel=None):
    from avengine.timeline.v7_motion_during_conditions import prepare_motion_during_row,verify_planned_trajectories
    counts=funnel if funnel is not None else Counter()
    actor=plan.actors[target];crop=float(slot["start_s"])-float(slot["local_start_s"])
    matching=[b for a,b in actor.metadata["walk_intervals_s"]
              if a<crop+10. and b>crop]
    end=max(matching)-crop-float(actor.metadata.get("ramp_s",0.))
    until=min(10.,end,slot["reply_local_s"]-1/15.)
    if len(slot.get("queries",[]))>=2:
        until=min(until,float(slot["queries"][1][0])+1/15.)
    tracks=emitter_tracks(plan.actors,assets);accepted=[];results={};failures={}
    for qa in FAMILIES:
        branch_value={"QA-06":"moving","QA-15":branch,"QA-16":None,"QA-17":"yes"}[qa]
        sub=deepcopy(row)
        sub.pop("motion_during_condition_plan",None)
        companion=next(a["asset_id"]for a in assets if a["asset_id"]!=target)
        sub["motion_source_start_s"]={target:slot["local_start_s"],companion:slot["reply_local_s"]}
        sub["motion_during_condition_requests"]=[{
            "qa_id":qa,"branch":branch_value,"target_asset_id":target,
            "speed_m_s":float(actor.metadata["speed_m_s"]),"speed_range_m_s":[.8,1.4],
            "end_hold_s":max(0.,10.-until),"offsets_s":[crop],"reserve_tail_s":.5,
            "wet_tail_planning_bound_s":.5,"minimum_distance_change_m":.5}]
        if qa in {"QA-16","QA-17"}:
            # Declare the existing predicate's single inter-frame step rather
            # than inventing a private .8/1.5-second post-motion requirement.
            sub["motion_during_condition_requests"][0]["minimum_motion_s"]=1/15.
        try:
            prepared=prepare_motion_during_row(sub,assets,records)
            proof=verify_planned_trajectories(prepared["motion_during_condition_plan"],
                                             tracks,plan.camera.position_m)
            passed=bool(proof.get(qa))
            if passed:accepted.extend(prepared["motion_during_condition_plan"]["plans"])
            else:
                from avengine.timeline.v7_motion_during_conditions import _cached_solution
                from avengine.rooms import conditioned_motion as cm
                details=[]
                for item in prepared["motion_during_condition_plan"]["plans"]:
                    at=crop+np.arange(150)/15.
                    positions={aid:np.column_stack([
                        np.interp(at,np.arange(len(track))/30.,track[:,axis])for axis in range(3)])
                        for aid,track in tracks.items()}
                    details.append(cm.verify_motion_candidate(_cached_solution(item["config"]),
                         positions_m=positions,root_positions_m=None,listener_position_m=plan.camera.position_m))
                failures[qa]={"original_verify_planned_trajectories":False,"reports":details}
        except ValueError as exc:
            passed=False;failures[qa]=str(exc)
        results[qa]=passed
        counts[qa+"_original_verified"if passed else qa+"_original_rejected"]+=1
    return accepted,results,failures


def combine_obligations(row,plan,plans,results):
    from avengine.rooms import conditioned_motion as cm
    row["motion_during_condition_plan"]={"status":"solved","plans":plans,
         "solver":cm.solver_signature(),"qa_predicates_changed":False,
         "owner":"core endpoint geometry plus unchanged original motion verifier"}
    row["human_motion_event_planned_verification"]=results
    plan.metadata["motion_condition_plan"]=row["motion_during_condition_plan"]
    plan.metadata["planned_emitter_positions_by_asset"]=emitter_tracks(plan.actors,
                                                plan.metadata["registered_assets"])
    return row
