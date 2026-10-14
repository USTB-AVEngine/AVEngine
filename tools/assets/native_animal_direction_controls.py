"""Actually rotate Habitat drawables backwards and sideways on the same route."""
import argparse,json,math
from pathlib import Path
import numpy as np
from avengine.capture.continuous_native_video import NativeContinuousVideo
from avengine.assets.animal_motion.native import instantiate,camera,overlay
from avengine.assets.animal_motion.checks import direction_check
from avengine.assets.roster_motion.contact import yaw_matrix

def main():
 p=argparse.ArgumentParser();p.add_argument('--motion',type=Path,nargs='+',required=True);p.add_argument('--house',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 h=json.loads(a.house.read_text());inputs=json.loads(a.runtime.read_text());first=json.loads(a.motion[0].read_text());eye=np.array(first['listener_m'])
 v=NativeContinuousVideo(h,{'listener_m':eye.tolist(),'camera_forward':first['camera_forward'],'sources':{}},inputs,[],width=640,height=480,source_material_cache_directory=a.output/'materials');results=[]
 try:
  for path in a.motion:
   m=json.loads(path.read_text());frames=m['frames'];velocity=(np.array(frames[-1]['root_world_m'])-frames[0]['root_world_m'])/(frames[-1]['time_s']-frames[0]['time_s']);yaw=math.atan2(-velocity[0],-velocity[2]);target=(np.array(frames[-1]['root_world_m'])+frames[0]['root_world_m'])/2+[0,.3,0];eye,cf,cr,cu=camera(v,eye,target)
   for degrees in (180,90):
    rows=[]
    for i in (0,len(frames)//2,len(frames)-1):
     f=frames[i];matrix=np.eye(4);matrix[:3,:3]=yaw_matrix(yaw+math.radians(degrees));matrix[:3,3]=f['root_world_m'];obj=instantiate(v,f['pose_glb'],matrix);hn=obj.root_scene_node.create_child();hn.translation=v.mn.Vector3(f['head_m']);tn=obj.root_scene_node.create_child();tn.translation=v.mn.Vector3(f['tail_m']);actual=np.asarray(obj.root_scene_node.absolute_transformation(),float)
     row={'time_s':f['time_s'],'root':actual[:3,3].tolist(),'head':np.asarray(hn.absolute_translation).tolist(),'tail':np.asarray(tn.absolute_translation).tolist()};rows.append(row)
     if i==len(frames)//2:
      rgb=np.asarray(v.sim.get_sensor_observations()['rgb'])[...,:3];image=a.output/(m['asset_id']+f'_{degrees}.png');overlay(rgb,eye,(cf,cr,cu),row['head'],row['tail'],actual[:3,3]+[0,.02,0],velocity).save(image);row['image']=str(image)
     v.sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)
    check=direction_check([r['time_s'] for r in rows],[r['root'] for r in rows],[r['head'] for r in rows],[r['tail'] for r in rows]);results.append({'asset_id':m['asset_id'],'body_rotation_error_deg':degrees,'frames':rows,'check':check,'physically_rotated_native_drawable':True})
  result={'status':'pass' if all(r['check']['status']=='fail' for r in results) else 'fail','controls':results,'excluded_from_admission':True};(a.output/'controls.json').write_text(json.dumps(result,indent=2));print(result['status'],len(results))
 finally:v.close()
if __name__=='__main__':main()
