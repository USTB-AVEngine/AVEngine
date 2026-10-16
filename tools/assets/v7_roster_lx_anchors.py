"""Primary LX sound anchors from manually reviewed native pixels and original triangles."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from v7_human_roster_materials import read_glb,json_write
from v7_human_roster_pose import pose,accessor
from v7_roster_contact_ik import yaw_matrix
ANNOTATIONS={'clapping_person':([301,297],'palms','clapping'),'flute_player':([350,266],'flute_open_end','flute'),'guitar_player':([296,309],'guitar_sound_hole','acoustic_guitar'),'hand_drum_player':([313,369],'drum_membrane','hand_drum'),'megaphone_speaker':([291,251],'megaphone_bell','amplified_speech'),'pianist':([329,335],'piano_keys_soundboard_region','piano'),'saxophone_player':([286,326],'saxophone_bell','saxophone'),'violin_player':([337,264],'violin_resonator','violin')}
def ray_tri(o,v,tri):
 e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];h=np.cross(np.broadcast_to(v,e2.shape),e2);det=np.einsum('ij,ij->i',e1,h);valid=abs(det)>1e-10;inv=np.where(valid,1/np.where(valid,det,1),0);s=o-tri[:,0];u=np.einsum('ij,ij->i',s,h)*inv;q=np.cross(s,e1);vv=q@v*inv;t=np.einsum('ij,ij->i',e2,q)*inv;valid&=(u>=0)&(vv>=0)&(u+vv<=1)&(t>0)
 if not valid.any():raise ValueError('Selected native sound pixel misses original asset triangles')
 return o+v*np.min(t[valid])
def run(T):
 view=json.loads((T/'evidence/habitat_view_mp3d.json').read_text());pv=np.asarray(view['rgb_projection_matrix'])@np.asarray(view['rgb_view_matrix']);inv=np.linalg.inv(pv);rot=yaw_matrix(view['yaw_rad']);root=np.asarray(view['object_position_m']);rows=[]
 for aid in sorted((T/'assets/runtime').glob('v7_lx_*')):
  meta=json.loads((aid/'original/poses.json').read_text());name=aid.name.replace('v7_lx_','');mouth=meta['poses']['stand']['mouth_m']
  if name in ANNOTATIONS:
   px,part,category=ANNOTATIONS[name];d,b=read_glb(aid/'original/habitat.glb');ps,ev=pose(d,b,canonical_flip=False,ground=False);tri=np.concatenate([(p['positions'])[accessor(d,b,d['meshes'][d['nodes'][p['node']]['mesh']]['primitives'][p['primitive']]['indices']).astype(int).reshape(-1,3)]for p in ps]);ndc=[2*(px[0]+.5)/640-1,1-2*(px[1]+.5)/640];a=inv@np.array([*ndc,-1,1]);z=inv@np.array([*ndc,1,1]);a=a[:3]/a[3];z=z[:3]/z[3];o=rot.T@(a-root);v=rot.T@(z-a);v/=np.linalg.norm(v);point=ray_tri(o,v,tri);status='pass_native_pixel_original_triangle_surface_medium_far'
  else:px=meta['poses']['stand'].get('mouth_annotation_pixel_mp3d');point=np.asarray(mouth);part='mouth';category=meta['sound_class'];status=meta['poses']['stand']['mouth_status']
  rec={'asset_id':aid.name,'primary_part':part,'primary_local_glb_m':point.tolist(),'primary_local_ue_cm':[-100*point[2],100*point[0],100*point[1]],'sound_class':category,'mouth_secondary_local_glb_m':mouth,'native_annotation_pixel_mp3d':px,'status':status,'method':'manually reviewed native MP3D image pixel and nearest source triangle; point anchor for medium/far scenes, not a precise radiation model','movement':'static','audio_admission_owner':'v7-families'};rows.append(rec)
  p=T/'round3/images/lx_emitters'/ (aid.name+'.jpg');p.parent.mkdir(parents=True,exist_ok=True);im=Image.open(T/'renders/habitat/mp3d'/aid.name/'original/rgb.png').convert('RGB');dr=ImageDraw.Draw(im);x,y=px;dr.ellipse((x-5,y-5,x+5,y+5),outline='lime',width=2);im.save(p,quality=85);rec['review_jpeg']=str(p)
  print('LX_PRIMARY',aid.name,part,point.tolist(),flush=True)
 json_write(T/'round3/registrations/lx_sound_anchors.json',rows)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
