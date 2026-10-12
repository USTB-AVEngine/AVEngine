"""Independent table/geometry reductions for the private integration pilot.

This evaluator never calls the STARSS or admitted-class question emitters.
Original v6 candidates use their retained authority replay separately.
"""
from collections import defaultdict
import math
import numpy as np

CLASS13=('female speech','male speech','clapping','telephone','laughter','domestic sounds','footsteps','door','music','musical instrument','water tap','bell','knock')
DIRECTIONS=('front','front-left','left','back-left','back','back-right','right','front-right')


def _segments(events,start):
    bins=defaultdict(set);endpoints=defaultdict(set);lo=round(start*10);hi=lo+100
    for e in events:
        key=(e['class_id'],e['source_id']);endpoints[key].add(e['endpoint_id'])
        bins[key].update(range(e['start_tick'],e['end_tick']))
    result=[]
    for (cls,actor),ticks in bins.items():
        if not ticks:continue
        ordered=sorted(ticks);first=last=ordered[0]
        for t in ordered[1:]+[ordered[-1]+2]:
            if t!=last+1:
                if first<hi and last+1>lo:result.append({'class_id':cls,'source_id':actor,'endpoint_id':next(iter(endpoints[(cls,actor)])) if len(endpoints[(cls,actor)])==1 else None,'start_tick':first,'end_tick':last+1,'anchor_tick':max(first,lo)-lo,'window_end_tick':min(last+1,hi)-lo})
                first=t
            last=t
    return sorted(result,key=lambda s:(s['anchor_tick'],s['class_id'],s['source_id']))


def _point(track,t):
    for k in track:
        if abs(k['time_s']-t)<1e-8:return np.asarray(k['position_m'],float)
    for a,b in zip(track,track[1:]):
        if a['time_s']<t<b['time_s'] and a['position_m']==b['position_m']:return np.asarray(a['position_m'],float)
    raise ValueError('independent replay has no observed geometry at query time')


def _geometry(scene,ep,t):
    f=np.asarray(scene['camera_forward'],float).copy();f/=np.linalg.norm(f);r=np.asarray(scene.get('camera_right',np.cross(f,[0.,1.,0.])));r/=np.linalg.norm(r);u=np.asarray(scene.get('camera_up',np.cross(r,f)));u/=np.linalg.norm(u)
    d=_point(scene['sources'][ep]['trajectory'],t)-np.asarray(scene['listener_m']);z=float(d@f)
    angle=(math.degrees(math.atan2(-float(d@r),z))+180)%360-180
    w,h=scene.get('resolution_wh',[1280,720]);cal=scene.get('camera_calibration',{});fx=float(cal.get('fx',w/(2*math.tan(math.radians(scene.get('hfov_deg',85)/2)))));fy=float(cal.get('fy',fx));shift=.5 if cal.get('coordinate_convention','zero_based_pixel_centers')=='zero_based_pixel_centers' else 0.
    cx=float(cal.get('cx',(w-1)/2))+shift;cy=float(cal.get('cy',(h-1)/2))+shift
    inside=z>1e-9 and 0<=cx+fx*float(d@r)/z<w and 0<=cy-fy*float(d@u)/z<h
    return angle,'in_frame' if inside else 'out_of_frame'


def validate_additional_items(public,private,proof):
    answers={r['question_id']:r for r in private};scene=proof['protocol_scene'];start=float(proof['window_start_s']);segments=_segments(scene['events'],start);checked=[];bad=[]
    class_tracks=defaultdict(set);all_tracks=defaultdict(set);lo=round(start*16000);hi=lo+160000
    for event in proof['events'].values():
        cls=event['sound_class'];ep=event['native_endpoint_id']
        for a,b in event['activity_intervals_samples_global']:
            a=max(lo,a);b=min(hi,b)
            if b>a:
                bins=set(range((a-lo)//1600,math.ceil((b-lo)/1600)));class_tracks[(cls,ep)].update(bins);all_tracks[ep].update(bins)
    for q in public:
        panel=q.get('protocol_panel');gold=answers[q['question_id']];family=q['qa_id']
        if panel=='STARSS23_v0.2':
            if family=='QA-23':expected=str(len(segments))
            elif family=='STARSS-SPEAKERS':expected=str(len({s['source_id'] for s in segments if s['class_id'] in {0,1}}))
            else:
                rank=int(gold['evidence']['candidate_rank']);s=segments[rank-1];ep=s['endpoint_id'];t=start+s['anchor_tick']/10
                if family=='STARSS-CLASS13':expected=CLASS13[s['class_id']]
                elif family=='QA-21':expected={8:'music_playback',10:'water_tap_faucet'}[s['class_id']]
                else:
                    angle,frustum=_geometry(scene,ep,t if family!='QA-25' else start+math.ceil(s['anchor_tick']/10))
                    if family=='STARSS-FRUSTUM':expected=frustum
                    elif family=='QA-04':expected=DIRECTIONS[math.floor((angle+22.5)/45)%8]
                    elif family=='QA-25':expected=str((math.floor(angle+.5)+180)%360-180)
                    else:raise ValueError('unhandled independent STARSS family '+family)
        elif panel=='v7_admitted_recording_classes':
            s=gold['evidence']['activity'];ep=s['endpoint'];a,b=s['start_tick'],s['end_tick'];matches=[]
            for (cls,key),ticks in class_tracks.items():
                if key==ep and all(t in ticks for t in range(a,b)) and a-1 not in ticks and b not in ticks:matches.append(cls)
            competing=sum(a in ticks and a-1 not in ticks for ticks in all_tracks.values())
            if len(matches)!=1 or competing!=1:raise ValueError('independent class target is ambiguous or not actual recorded activity')
            expected=matches[0]
        else:continue
        form_answer=expected if q['form']=='open' else next(c['id'] for c in q['choices'] if c['text']==expected)
        ok=expected==str(gold['canonical_answer']) and form_answer==str(gold['answer']);item={'question_id':q['question_id'],'qa_id':family,'status':'pass' if ok else 'fail','expected_canonical':expected,'expected_form_answer':form_answer};checked.append(item)
        if not ok:bad.append({**item,'actual_canonical':gold['canonical_answer'],'actual_form_answer':gold['answer']})
    return {'method':'independent 100ms union / source-identity sets / actual recording activity / native camera projection; no panel emitters called','checked_forms':len(checked),'passed_forms':sum(r['status']=='pass' for r in checked),'mismatches':bad,'items':checked,'status':'fail' if bad else 'pass' if checked else 'not_run'}
