"""Bake paired navigation paths and contact-corrected Rocketbox walking clips."""
import argparse,copy,json,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from v7_human_roster_materials import read_glb,write_glb,json_write
from v7_human_roster_pose import animation_times,globals_at,rigid_glb,pose
from v7_human_roster_ue_root import promote
from v7_roster_contact_ik import FootPlant,find_bone,C,serialize_action
from v7_roster_floor_grid import FloorGrid
def play_rate(d,b,ac):
 ts=animation_times(d,b,ac);g=[globals_at(d,b,ac,float(t))for t in ts];vals=[]
 for side in ['L','R']:
  node=find_bone(d,' '+side+' Foot');xyz=np.stack([(C@x[node])[:3,3]for x in g]);v=np.gradient(xyz[:,2],ts);vals.extend(v[v>.4].tolist())
 speed=float(np.median(vals));return 1.2/speed,speed
def one(args):
 T,aid,houses=args;T=Path(T);R=T/'round2';D=R/'assets/humans'/aid;d,b=read_glb(D/'final_ue_root.glb');base_d,base_b=copy.deepcopy(d),bytes(b);ac=next(x for x in base_d['animations']if x['name']=='Walking');ts=animation_times(base_d,base_b,ac);rate,nominal=play_rate(base_d,base_b,ac);records=[]
 for house in houses:
  path_source=R/'registrations'/('path_kujiale_v3.json'if house=='kujiale'else'path_'+house+'.json');path=json.loads(path_source.read_text())
  if aid in ['v7_fire_female_01','v7_police_female_01']:
   oldt=np.asarray(path['time_s']);oldp=np.asarray(path['points_m']);n=int(math.ceil(path['length_m']/1.05*30));newt=np.arange(n+1)/30;phase=np.linspace(oldt[0],oldt[-1],n+1);points=np.column_stack([np.interp(phase,oldt,oldp[:,j])for j in range(3)]);yaw=np.interp(phase,oldt,np.unwrap(path['yaw_rad']));speed=np.linalg.norm(np.diff(points[:,[0,2]],axis=0),axis=1)*30
   path.update(time_s=newt.tolist(),points_m=points.tolist(),yaw_rad=yaw.tolist(),speed_m_s=speed.tolist(),duration_s=float(newt[-1]),max_turn_speed_deg_s=float(np.rad2deg(abs(np.diff(yaw))*30).max()))
   path_source=T/'round3/registrations/paths_by_actor'/aid/(house+'.json');json_write(path_source,path)
  localrate=float(np.mean(path['speed_m_s']))/nominal
  floor=FloorGrid(R/'evidence'/('floor_grid_'+house+('_v3'if house=='kujiale'else'')+'.npz'));times=path['time_s'];pl=FootPlant(base_d,base_b,ac,times,path['points_m'],path['yaw_rad'],localrate,ts[0],ts[-1]-ts[0],floor);frames=[];gs=[];vertices=[]
  for i,t in enumerate(times):
   g,parts,ev=pl.frame(i);gs.append(g);frames.append(ev);geometry=np.concatenate([p['positions']for p in parts]);vertices.append(geometry)
  name='Walking_Contact_'+house;d['animations']=[a for a in d['animations']if a['name']!=name];d,b=serialize_action(d,b,name,times,gs)
  contacts=[f for e in frames for f in e['feet'].values()if f['in_contact']];lo=min(f['floor_delta_m']for e in frames for f in e['feet'].values());hi=max(abs(f['floor_delta_m'])for f in contacts);lock=max(f['support_lock_error_m']for f in contacts);lower=max(e['pelvis_lowering_m']for e in frames);maxradius=max(float(np.linalg.norm(v[:,[0,2]],axis=1).max())for v in vertices);maxheight=max(float(v[:,1].max())for v in vertices)
  segments={};events=[]
  for fi,e in enumerate(frames):
   for side,f in e['feet'].items():
    if not f['in_contact']:continue
    key=(side,f['support_segment_start_frame'],tuple(f['support_vertex']));segments.setdefault(key,[]).append(np.asarray(f['support_world_m']))
    if f['support_segment_start_frame']==fi:events.append({'time_s':e['time_s'],'side':side,'world_support_m':f['support_world_m'],'kind':'sole_support_change','footstep_audio_generated':False})
  drift=max(float(np.linalg.norm((np.stack(v)-v[0])[:,[0,2]],axis=1).max())for v in segments.values());continuous_segments=sum(len(v)>=3 for v in segments.values());jumps=max(float(np.linalg.norm(np.asarray(frames[i]['feet'][s]['ankle_world_m'])-np.asarray(frames[i-1]['feet'][s]['ankle_world_m'])))for i in range(1,len(frames))for s in ['L','R'])
  passed=lo>=-.008 and hi<.008 and lock<.003 and drift<.003 and pl.max_reach<.001 and lower<.08 and jumps<.15 and continuous_segments>=2
  rec={'asset_id':aid,'house':house,'clip':name,'source_clip':'Walking','frame_count':len(frames),'sample_rate_hz':30,'duration_s':float(times[-1]),'source_play_rate':localrate,'path_manifest':str(path_source),'coordinate_revision':path.get('coordinate_revision','native_room'),'source_nominal_stride_speed_m_s':nominal,'root_speed_min_m_s':min(path['speed_m_s']),'root_speed_max_m_s':max(path['speed_m_s']),'max_turn_speed_deg_s':path['max_turn_speed_deg_s'],'minimum_foot_floor_delta_m':lo,'maximum_contact_floor_delta_m':hi,'maximum_support_lock_error_m':lock,'maximum_stance_vertex_horizontal_drift_m':drift,'maximum_ik_reach_error_m':pl.max_reach,'maximum_pelvis_lowering_m':lower,'maximum_ankle_frame_step_m':jumps,'sustained_support_segments':continuous_segments,'maximum_actor_vertex_horizontal_radius_m':maxradius,'maximum_actor_vertex_height_m':maxheight,'source_navmesh_agent_radius_m':path['agent_radius_m'],'source_navmesh_agent_height_m':path['agent_height_m'],'footstep_audio_status':'not_applicable_no_audio_generated; export support events for v7-families','native_collision_status':'pending_native_scene_sweep','native_dynamic_foa_status':'pending_native_readback','geometric_contact_status':'pass'if passed else'fail','frames':frames,'support_events':events}
  json_write(D/('walking_'+house+'.json'),rec);records.append({k:v for k,v in rec.items()if k not in ['frames','support_events']})
  # Check the serialized channel values by reading them back, rather than trusting the writer.
  ca=d['animations'][-1];errors=[]
  for fi in [0,len(times)//2,len(times)-1]:
   _,ev=pose(d,b,ca,float(times[fi]),ground=False);errors.append(float(np.linalg.norm(np.asarray(ev['mouth_m'])-np.asarray(frames[fi]['mouth_m']))))
  if max(errors)>.0001:raise RuntimeError('Contact clip readback mouth disagreement')
  print('WALK_CONTACT',aid,house,rec['geometric_contact_status'],round(hi,5),round(drift,6),flush=True)
 nd,nb,promotion=promote(d,b)
 from v7_roster_disk import check
 check(T,64*1024**2);temporary=D/'final_ue_root.new.glb';write_glb(temporary,nd,nb);temporary.replace(D/'final_ue_root.glb');json_write(D/'walking_summary.json',{'asset_id':aid,'paths':records,'ue_root_promotion':promotion})
 return records
def main(a):
 T=a.task_root;rows=json.loads((T/'registrations/selection.json').read_text())['avatars']
 if a.only:rows=[r for r in rows if r['asset_id']==a.only]
 with ProcessPoolExecutor(max_workers=min(a.workers,3))as pool:out=[r for records in pool.map(one,[(str(T),r['asset_id'],a.house or ['kujiale'])for r in rows])for r in records]
 json_write(T/'round3/evidence/walking_recompiled_results.json',out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--only');p.add_argument('--house',action='append',choices=['mp3d','hm3d','kujiale']);p.add_argument('--workers',type=int,default=3);main(p.parse_args())
