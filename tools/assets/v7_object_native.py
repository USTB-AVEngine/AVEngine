"""Admit static objects on raycast stage supports in MP3D and HM3D with native PBR, geometry and colour metrics."""
from v7_object_progress import append as progress
import argparse,json,math,sys,traceback,os,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from avengine.assets.source_materials import configure_source_pbr,measure_source_appearance,source_palette,read_glb
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime,_make_configuration
from avengine.rooms.contracts import load_and_validate_inputs
from v7_human_roster_habitat import camera,project
from v7_object_prepare import dump
from measure_static_upright_correction import load_glb
from v7_object_color import measure as measure_color,summarize_views

def rotation(yaw):
    c,s=math.cos(yaw),math.sin(yaw);return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def ray(sim,r,p,d,length,exclude=None):
    hits=sim.cast_ray(r.habitat_sim.geo.Ray(r.magnum.Vector3(np.asarray(p,dtype=float)),r.magnum.Vector3(np.asarray(d,dtype=float))),length).hits
    hits=[h for h in hits if h.object_id!=exclude]
    return None if not hits else np.asarray(hits[0].point,dtype=float)
def add(sim,r,path,root,yaw,sid):
    manager=sim.get_object_template_manager();handle=str(path)+'.v7_margin_1mm';t=manager.create_template(handle,register_template=False);t.render_asset_handle=str(path);t.collision_asset_handle=str(path);t.compute_COM_from_shape=False;t.shader_type='pbr';t.semantic_id=sid;t.margin=.001;manager.register_template(t,handle)
    obj=sim.get_rigid_object_manager().add_object_by_template_handle(handle)
    if obj is None:raise RuntimeError('Cannot load owned source object')
    obj.motion_type=r.habitat_sim.physics.MotionType.KINEMATIC;obj.translation=r.magnum.Vector3(root);obj.rotation=r.magnum.Quaternion.rotation(r.magnum.Rad(yaw),r.magnum.Vector3.y_axis());obj.semantic_id=sid;obj.root_scene_node.semantic_id=sid;obj.override_collision_group(r.habitat_sim.physics.CollisionGroups.Dynamic);return obj

def semantic_anchors(sim):
    result={'door':[],'window':[]}
    for obj in getattr(sim.semantic_scene,'objects',[]) or []:
        if obj is None:continue
        try:
            category=obj.category.name().lower();center=np.asarray(obj.aabb.center,float);size=np.asarray(obj.aabb.sizes,float)
            key='door' if 'door' in category else ('window' if 'window' in category else None)
            if key is None or not np.isfinite(center).all():continue
            if key=='door' and not (1.45<=size[1]<=3.0 and max(size[0],size[2])>=.45):continue
            result[key].append({'kind':key,'authority':'native semantic object AABB','native_object_id':str(obj.id),'category':category,'center_m':center.tolist(),'sizes_m':size.tolist()})
        except Exception:continue
    return result

def anchor_distance(point,anchor):
    center=np.asarray(anchor['center_m']);half=np.asarray(anchor.get('sizes_m',[0,0,0]))/2;delta=np.maximum(np.abs(np.asarray(point)-center)-half,0);return float(np.linalg.norm(delta[[0,2]]))

def geometric_doorways(sim,r,walls):
    # An actual rectangular room-entry opening: clear at two body heights,
    # at least 40cm wide, and a native lintel overhead. No inserted doorway.
    found=[];seen=set()
    for wall in walls[:256]:
        p=np.asarray(wall['point']);n=np.asarray(wall['normal']);t=np.cross(n,[0,1,0]);fy=wall['floor_y']
        for offset in [-1.2,-.8,-.4,.4,.8,1.2]:
            center=p+t*offset;key=tuple(np.round(center[[0,2]]/.2).astype(int))
            if key in seen:continue
            seen.add(key);clear=True
            for lateral in [-.2,0,.2]:
                for height in [.8,1.65]:
                    origin=center+n*.35+t*lateral;origin[1]=fy+height;hit=ray(sim,r,origin,-n,1.8)
                    if hit is not None and float((origin-hit)@n)<.85:clear=False;break
                if not clear:break
            if not clear:continue
            origin=center+n*.35;origin[1]=fy+2.25;lintel=ray(sim,r,origin,-n,.65)
            if lintel is None:continue
            center[1]=fy+1.05;found.append({'kind':'door','authority':'native raycast rectangular room-entry doorway: clear 0.4m width at 0.8m/1.65m and solid lintel at 2.25m','center_m':center.tolist(),'sizes_m':[.0,2.1,.0],'normal':n.tolist(),'minimum_opening_width_m':.4})
    return found

def survey(sim,r,view):
    center=np.asarray(view['object_position'],float);points=[];anchors=semantic_anchors(sim);seeds=[]
    for dx in np.arange(-2.4,2.401,.3):
        for dz in np.arange(-2.4,2.401,.3):seeds.append(center+np.array([dx,0,dz]))
    # Include local native doorway/window regions rather than a generic wall alone.
    for anchor in anchors['door']+anchors['window']:
        ac=np.asarray(anchor['center_m']);size=np.asarray(anchor['sizes_m'])
        if np.linalg.norm(ac[[0,2]]-center[[0,2]])>7.:continue
        ac[1]-=size[1]/2
        for dx in [-1.2,-.6,0,.6,1.2]:
            for dz in [-1.2,-.6,0,.6,1.2]:seeds.append(ac+np.array([dx,.1,dz]))
    for seed in seeds:
        snap=np.asarray(sim.pathfinder.snap_point(seed))
        if not np.isfinite(snap).all() or np.linalg.norm(snap[[0,2]]-seed[[0,2]])>.4:continue
        floor=ray(sim,r,snap+[0,.2,0],[0,-1,0],.6)
        if floor is not None:points.append(floor)
    points.sort(key=lambda p:float(np.linalg.norm(p-center)));tables=[];walls=[];high=[];window=[];ceilings=[];floor_y=float(np.median([p[1] for p in points])) if points else float(center[1])
    for dx in np.arange(-3.0,3.001,.15):
        for dz in np.arange(-3.0,3.001,.15):
            start=np.array([center[0]+dx,floor_y+1.45,center[2]+dz]);top=ray(sim,r,start,[0,-1,0],1.4)
            if top is not None and .50<=top[1]-floor_y<=1.22:tables.append({'point':top.tolist(),'floor_y':floor_y,'normal':[0,1,0],'height_above_floor_m':float(top[1]-floor_y),'surface':'table'})
    tables.sort(key=lambda x:float(np.linalg.norm(np.asarray(x['point'])[[0,2]]-center[[0,2]])))
    for floor in points:
        ceil=ray(sim,r,floor+[0,1.8,0],[0,1,0],2.3)
        if ceil is not None and 2.15<=ceil[1]-floor[1]<=3.9:ceilings.append({'point':ceil.tolist(),'floor_y':float(floor[1]),'normal':[0,-1,0],'surface':'ceiling'})
        for height in [1.45,2.2,2.4,1.15,1.8]:
            for d in [[1,0,0],[-1,0,0],[0,0,1],[0,0,-1]]:
                p=floor+[0,height,0];hit=ray(sim,r,p,d,2.7)
                if hit is None or np.linalg.norm(hit-p)<.25:continue
                base={'point':hit.tolist(),'floor_y':float(floor[1]),'normal':(-np.asarray(d)).tolist(),'surface':'wall','height_above_floor_m':height}
                if height==1.45:walls.append(base)
                if height>=2.2:high.append(dict(base,placement_policy='native high wall; attachment centre >=2.2m, object bottom >=1.9m above floor'))
                if height<=1.8:
                    close=[a for a in anchors['window'] if anchor_distance(hit,a)<=.35 and abs(hit[1]-a['center_m'][1])<=max(.5,a['sizes_m'][1]/2)]
                    if close:window.append(dict(base,native_window_anchor=min(close,key=lambda a:anchor_distance(hit,a)),placement_policy='native window AABB frame region within 0.35m'))
    if not anchors['door']:anchors['door']=geometric_doorways(sim,r,walls)
    door=[]
    for wall in walls:
        close=[a for a in anchors['door'] if anchor_distance(wall['point'],a)<=1.2]
        if close:
            anchor=min(close,key=lambda a:anchor_distance(wall['point'],a));door.append(dict(wall,native_door_anchor=anchor,distance_to_native_door_aabb_m=anchor_distance(wall['point'],anchor),placement_policy='native wall within 1.2m of a native door or measured room-entry doorway; whole-building entrance identity unverified'))
    floors=[{'point':p.tolist(),'normal':[0,1,0],'surface':'floor','floor_y':float(p[1])} for p in points]
    return {'floor':[dict(p,yaw_rad=y) for p in floors for y in [0.,math.pi/2,math.pi,-math.pi/2]],'table':[dict(p,yaw_rad=y) for p in tables for y in [0.,math.pi/2,math.pi,-math.pi/2]],'wall':walls,'high_wall':high,'window_wall':window,'doorbell_wall':door,'native_semantic_anchors':anchors,'ceiling':[dict(p,yaw_rad=y) for p in ceilings for y in [0.,math.pi/2,math.pi,-math.pi/2]]}

def attachment_template(meta):
    vertices,_=load_glb(Path(meta['derived_path']));surface=meta['attachment_surface'];axis=0 if surface=='wall' else 1;high=surface=='ceiling';values=vertices[:,axis];extreme=float(values.max() if high else values.min());band=min(.002,.05*float(np.ptp(values)));contact=vertices[(extreme-values if high else values-extreme)<=band+1e-9];axes=tuple(x for x in range(3) if x!=axis);low=contact.min(axis=0);upper=contact.max(axis=0);anchor=(low+upper)/2;anchor[axis]=extreme;span=upper-low;local=[];chosen=set()
    for da,db in [(0,0),(-.40,-.40),(-.40,.40),(.40,-.40),(.40,.40)]:
        target=anchor.copy();target[axes[0]]+=da*span[axes[0]];target[axes[1]]+=db*span[axes[1]];order=np.argsort(np.sum((contact-target)**2,axis=1));index=next((int(i) for i in order if int(i) not in chosen),int(order[0]));chosen.add(index);local.append(contact[index])
    return anchor,local,{'definition':'actual GLB vertices in the outer <=2mm attachment band; excludes non-contact television panel overhang','axis':axis,'outer_band_m':band,'contact_vertices':len(contact),'contact_footprint_min_m':low.tolist(),'contact_footprint_max_m':upper.tolist(),'probe_local_m':[v.tolist() for v in local],'distinct_probe_vertices':len(chosen)}

def support_points(meta,support):
    normal=np.asarray(support['normal']);point=np.asarray(support['point']);yaw=math.atan2(-normal[2],normal[0]) if meta['attachment_surface']=='wall' else support.get('yaw_rad',0.);R=rotation(yaw);anchor,local,_=meta['_attachment_template'];root=point-R@anchor
    return root,yaw,normal,local

def check_support(sim,r,meta,support):
    root,yaw,normal,local=support_points(meta,support);R=rotation(yaw);deltas=[];hits=[]
    for q in local:
        p=root+R@q;h=ray(sim,r,p+normal*.10,-normal,.22)
        if h is None:return None
        deltas.append(float((p-h)@normal));hits.append(h.tolist())
    if max(abs(x) for x in deltas)>.015:return None
    return {'root_m':root.tolist(),'yaw_rad':yaw,'normal':normal.tolist(),'contact_deltas_m':deltas,'support_raycast_points_m':hits,'support':support,'threshold_m':.015,'actual_mesh_attachment':meta['_attachment_template'][2],'support_identification':'native stage horizontal planar patch at table height, or declared floor/wall/ceiling; no inserted support geometry'}

def functional_exclusion(meta,q,front_facing=None):
    x,y,z=np.asarray(q).T;k=meta['object_type'];form=meta.get('form_factor','');exclude=np.zeros(len(x),bool)
    # Inset functional surfaces can lie well behind the foremost bezel: x>0.8
    # incorrectly left entire clock dials/grilles in the main-housing sample.
    if k=='cellphone' and meta.get('functional_front_axis')=='positive-y':exclude=(y>.10)&(x>.08)&(x<.92)&(z>.08)&(z<.92)
    elif k in ['television','cellphone','tablet']:exclude=(x>.10)&(y>.13)&(y<.94)&(z>.075)&(z<.925)
    elif k=='microwave_oven':exclude=(x>.10)&(y>.12)&(y<.90)&(z>.08)&(z<.92)
    elif k=='wall_clock':
        inside=((y-.5)/.42)**2+((z-.5)/.42)**2<1 if 'round' in form else ((y>.12)&(y<.88)&(z>.12)&(z<.88))
        exclude=(x>.08)&inside
    elif k=='mantel_clock':
        inside=((y-.55)/.36)**2+((z-.5)/.41)**2<1 if 'round' in form else ((y>.22)&(y<.88)&(z>.13)&(z<.87))
        exclude=(x>.08)&inside
    elif k=='alarm_clock':
        inside=((y-.40)/.29)**2+((z-.5)/.41)**2<1 if 'twin_bell' in form else ((y>.13)&(y<.87)&(z>.10)&(z<.90))
        exclude=(x>.10)&inside
    elif k in ['bookshelf_speaker','floorstanding_speaker','soundbar','radio']:exclude=(x>.10)&(y>.04)&(y<.96)&(z>.04)&(z<.96)
    elif k=='blender':exclude=y>.44
    elif k=='printer':exclude=(x>.10)&(y>.55)&(y<.85)&(z>.60)&(z<.90)
    elif k=='desk_telephone':exclude=(x>.10)&(y>.12)&(y<.60)&(z>.20)&(z<.80)
    elif k=='landline_phone':exclude=(x>.10)&(y>.18)&(y<.78)&(z>.55)&(z<.94)
    elif k=='security_keypad' and 'flip_closed' in form:exclude=np.zeros(len(x),bool) # opaque closed casing lid is body, not exposed buttons
    elif k=='security_keypad':exclude=(x>.10)&(y>.22)&(y<.85)&(z>.20)&(z<.80)
    elif k=='doorbell_chime_unit':exclude=(x>.10)&(y>.25)&(y<.85)&(z>.3)&(z<.7)
    elif k in ['smoke_detector','carbon_monoxide_alarm','kitchen_timer']:exclude=(x>.10)&(y>.35)&(y<.65)&(z>.35)&(z<.65)
    if front_facing is not None and k!='blender':exclude &= np.asarray(front_facing,dtype=bool)
    return exclude

def body_mask(rgb,semantic,depth,sim,r,uuid,root,yaw,meta):
    sensor=sim._sensors[uuid]._sensor_object
    P=np.asarray(sensor.render_camera.projection_matrix)
    W=np.asarray(sensor.node.absolute_transformation())
    return body_mask_from_native_depth(semantic,depth,P,W,root,yaw,meta)

def body_mask_from_native_depth(semantic,depth,P,W,root,yaw,meta):
    yy,xx=np.where(semantic);mask=semantic.copy()
    if not len(xx):return mask
    P=np.asarray(P);W=np.asarray(W);h,w=semantic.shape
    z=np.asarray(depth)[yy,xx].reshape(-1);vx=((xx+.5)/w*2-1)*z/P[0,0];vy=(1-(yy+.5)/h*2)*z/P[1,1];cam=np.column_stack([vx,vy,-z,np.ones(len(z))]);world=cam@W.T;local=(world[:,:3]-np.asarray(root))@rotation(yaw);bbox=np.asarray(meta['geometry']['bbox_m']);q=(local-bbox[0])/np.maximum(bbox[1]-bbox[0],1e-6);x,y,z=q.T;k=meta['object_type']
    # A kickstand changes the X bounding box: the coloured back of a tablet
    # can occupy the same normalized X range as its screen. Reconstruct outward
    # surface normals from native depth; only front-facing functional surfaces
    # may be excluded. This also retains curved side/back housings.
    grid=np.full((h,w,3),np.nan,dtype=float);grid[yy,xx]=local
    dx=np.roll(grid,-1,axis=1)-np.roll(grid,1,axis=1);dy=np.roll(grid,-1,axis=0)-np.roll(grid,1,axis=0)
    n=np.cross(dx[yy,xx],dy[yy,xx]);length=np.linalg.norm(n,axis=1);n/=np.maximum(length[:,None],1e-12)
    local_eye=(W[:3,3]-np.asarray(root))@rotation(yaw)
    n[np.sum(n*(local_eye-local),axis=1)<0]*=-1
    front_axis=1 if meta.get('functional_front_axis')=='positive-y' else 0
    front_facing=np.isfinite(n).all(axis=1)&(n[:,front_axis]>.35)
    exclude=functional_exclusion(meta,q,front_facing)
    mask[yy[exclude],xx[exclude]]=False
    # One-pixel interior avoids antialiasing/background mixing.
    from scipy.ndimage import binary_erosion
    mask=binary_erosion(mask,iterations=1)
    return mask

def render_one(sim,r,uuids,meta,supports,out,thresholds,scene,shader):
    sid=6500011;path=Path(meta['derived_path']);bbox=np.asarray(meta['geometry']['bbox_m']);dims=bbox[1]-bbox[0];surface=meta['attachment_surface'];D=out/meta['asset_id'];D.mkdir(parents=True,exist_ok=True);chosen=None;obj=None
    meta['_attachment_template']=attachment_template(meta)
    support_key='doorbell_wall' if meta['object_type']=='doorbell_chime_unit' else ('high_wall' if meta['object_type']=='air_conditioner' and meta['form_factor']=='wall_split' else ('window_wall' if meta['object_type']=='air_conditioner' and meta['form_factor']=='window_unit' else surface))
    # Reuse every stage candidate; require an exact patch and owner-specific location.
    for support in supports[support_key]:
        placement=check_support(sim,r,meta,support)
        if placement is None:continue
        if support_key=='high_wall' and placement['root_m'][1]+float(bbox[0,1])-support['floor_y']<1.9:continue
        root=np.asarray(placement['root_m']);yaw=placement['yaw_rad'];normal=np.asarray(placement['normal']);obj=add(sim,r,path,root,yaw,sid)
        # Reject an obstructed placement before rendering; search another exact stage support.
        obj.translation=r.magnum.Vector3(root+normal*.06);obstructed=bool(obj.contact_test());obj.translation=r.magnum.Vector3(root-normal*.03);positive_control=bool(obj.contact_test());obj.translation=r.magnum.Vector3(root)
        if obstructed or not positive_control:
            sim.get_rigid_object_manager().remove_object_by_id(obj.object_id);obj=None;continue
        R=rotation(yaw);target=root+R@np.mean(bbox,axis=0);d=max(.15,float(max(dims)))*1.45
        views=[]
        angles=[('front',0),('quarter_left',-math.pi/6),('quarter_right',math.pi/6),('side',math.pi/3)] if surface in ['wall','ceiling'] else [('front',0),('quarter_left',-math.pi/4),('quarter_right',math.pi/4),('back',math.pi)]
        for name,angle in angles:
            accepted=None
            for factor in [1.,1.3,.82,1.7]:
                if accepted:break
                offset=R@np.array([math.cos(angle)*d*factor,.18*d*factor,-math.sin(angle)*d*factor])
                if surface=='ceiling':offset[1]=-.9*d*factor
                eye=target+offset
                try:sim.initialize_agent(0,camera(r,eye,target));obs=sim.get_sensor_observations()
                except Exception:continue
                mask=np.asarray(obs[uuids['semantic']])==sid;yy,xx=np.where(mask)
                if len(xx)>700 and min(xx.min(),yy.min(),639-xx.max(),639-yy.max())>=3:accepted=(eye,obs,mask);break
            if accepted is None:break
            eye,obs,mask=accepted;rgb=np.asarray(obs[uuids['rgb']])[...,:3];dep=np.asarray(obs[uuids['depth']]);body=body_mask(rgb,mask,dep,sim,r,uuids['rgb'],root,yaw,meta)
            doc,blob=read_glb(path);palette=source_palette(doc,blob,path);highlight=measure_source_appearance(rgb,mask,palette=palette,max_highlight_ratio=.08,strict=False,metallic_surface=meta['object_type']=='floor_drain');saturated=float(np.mean(np.all(rgb[mask]>=245,axis=1)))
            color=measure_color(rgb,body,thresholds);ip=D/(name+'.jpg');Image.fromarray(rgb).save(ip,quality=88);Image.fromarray(rgb).save(D/(name+'.png'));Image.fromarray(mask.astype('uint8')*255).save(D/(name+'_mask.png'));Image.fromarray(body.astype('uint8')*255).save(D/(name+'_body.png'));np.save(D/(name+'_depth.npy'),dep)
            views.append({'name':name,'rgb':str(ip),'rgb_lossless':str(D/(name+'.png')),'mask':str(D/(name+'_mask.png')),'body_mask':str(D/(name+'_body.png')),'depth':str(D/(name+'_depth.npy')),'eye_m':eye.tolist(),'camera_projection':np.asarray(sim._sensors[uuids['rgb']]._sensor_object.render_camera.projection_matrix).tolist(),'camera_world_from_sensor':np.asarray(sim._sensors[uuids['rgb']]._sensor_object.node.absolute_transformation()).tolist(),'visible_pixels':int(mask.sum()),'border_margin_px':int(min(xx.min(),yy.min(),639-xx.max(),639-yy.max())),'color':color,'production_appearance':highlight,'saturated_ratio_rgb_ge245':saturated,'highlights_status':'pass' if highlight.get('highlight_ratio',highlight.get('specular_highlight_ratio',1))<=.08 else 'fail'})
        if len(views)==4:chosen=(placement,views,obj);break
        sim.get_rigid_object_manager().remove_object_by_id(obj.object_id);obj=None
    if chosen is None:
        rec={'asset_id':meta['asset_id'],'review_method_version':'object-attachment-body-v5','derived_glb':str(path),'derived_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':'fail','reasons':['no_native_'+support_key+'_support_with_complete_views'],'candidate_count':len(supports[support_key]),'scene':str(scene)};dump(D/'result.json',rec);return rec
    placement,views,obj=chosen;root=np.asarray(placement['root_m']);yaw=placement['yaw_rad'];normal=np.asarray(placement['normal']);native=None;err=None;follow=None
    if not meta.get('silent_decor'):
        local=np.asarray(meta['emitter']['local_m']);child=obj.root_scene_node.create_child();child.translation=r.magnum.Vector3(local);native=np.asarray(child.absolute_translation);err=float(np.linalg.norm(native-root-rotation(yaw)@local));delta=np.array([.02,.01,-.03]);newyaw=yaw+.2;obj.translation=r.magnum.Vector3(root+delta);obj.rotation=r.magnum.Quaternion.rotation(r.magnum.Rad(newyaw),r.magnum.Vector3.y_axis());follow=float(np.linalg.norm(np.asarray(child.absolute_translation)-(root+delta+rotation(newyaw)@local)));obj.translation=r.magnum.Vector3(root);obj.rotation=r.magnum.Quaternion.rotation(r.magnum.Rad(yaw),r.magnum.Vector3.y_axis())
    at=bool(obj.contact_test());obj.translation=r.magnum.Vector3(root-normal*.03);buried=bool(obj.contact_test());obj.translation=r.magnum.Vector3(root+normal*.06);raised=bool(obj.contact_test());obj.translation=r.magnum.Vector3(root)
    reasons=[]
    if not buried:reasons.append('native_buried_contact_control_failed')
    if raised:reasons.append('collision_after_moving_6cm_off_support')
    if not meta.get('silent_decor') and (err>.0001 or follow>.0001):reasons.append('emitter_native_follow_failed')
    if any(v['production_appearance'].get('highlight_ratio',1)>.08 for v in views):reasons.append('excess_highlights')
    pooled=summarize_views([v['color'] for v in views],thresholds);label=pooled['label']
    if label is None:reasons.append('no_dominant_body_color')
    rec={'asset_id':meta['asset_id'],'review_method_version':'object-attachment-body-v5','status':'pass' if not reasons else 'fail','reasons':reasons,'scene':str(scene),'placement':placement,'derived_glb':str(path),'derived_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_pose':meta['geometry']['resting_pose'],'shader':shader,'screenshots':views,'color_label':label,'main_color_share':pooled['main_share'],'pooled_color':pooled,'emitter_policy':'not_applicable_silent_decor' if meta.get('silent_decor') else 'native_scene_child_follow','emitter_world_m':None if native is None else native.tolist(),'emitter_native_readback_error_m':err,'emitter_translation_yaw_follow_error_m':follow,'contact_at_attachment':at,'buried_3cm_positive_contact':buried,'moved_6cm_off_support_contact':raised,'physics':'Bullet actual stage + owned kinematic Dynamic group override; 1mm margin','audio':'not_run per task scope','body_roi':'v5 native semantic/depth outward normals distinguish functional front from housing sides/back; geometric inset dial/screen/grille exclusion; coloured telephone handset retained; front/quarters/back for floor/table, front/quarters/side for mounts; 1px erosion'}
    dump(D/'result.json',rec);sim.get_rigid_object_manager().remove_object_by_id(obj.object_id);return rec

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--views',type=Path,required=True);p.add_argument('--only');p.add_argument('--house');p.add_argument('--output-name',default='native_existing');a=p.parse_args();T=a.task_root;os.environ['__GL_SHADER_DISK_CACHE_PATH']=str(T/'cache/gl');cal=json.loads((T/'color_calibration.json').read_text());assert cal['status']=='pass';metas=json.loads(a.manifest.read_text());metas=[m for m in metas if 'derived_path' in m and (not a.only or a.only in m['asset_id'])];records=[]
    for view in json.loads(a.views.read_text()):
        house=view['house']
        if a.house and house!=a.house:continue
        r=prepare_installed_habitat_runtime(runtime_prefix='/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z',magnum_python_site='/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages',mp3d_root=view['data_root'],rlr_sdk_root='/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg')
        inputs=load_and_validate_inputs(Path(view['room_manifest']),Path(view['capture_request']));cfg,uuids,_,scene=_make_configuration(inputs,None,T/'renders'/a.output_name/house,mp3d_root=r.mp3d_root,include_audio_sensor=False,physics_config_path=r.physics_config_path);cfg.sim_cfg.enable_physics=True;cfg.sim_cfg.gpu_device_id=0
        for ag in cfg.agents:
            for spec in ag.sensor_specifications:spec.resolution=[640,640]
        with r.habitat_sim.Simulator(cfg) as sim:
            shader=configure_source_pbr(cfg,r.habitat_sim,metadata_mediator=sim.metadata_mediator);supports=survey(sim,r,view);dump(T/(a.output_name+'_supports_'+house+'.json'),supports);print('SUPPORTS',house,{k:len(v) for k,v in supports.items() if isinstance(v,list)},flush=True)
            for meta in metas:
                try:rec=render_one(sim,r,uuids,meta,supports,T/'renders'/a.output_name/house,cal['thresholds'],scene,shader)
                except Exception as e:traceback.print_exc();rec={'asset_id':meta['asset_id'],'review_method_version':'object-attachment-body-v5','derived_glb':meta['derived_path'],'derived_sha256':hashlib.sha256(Path(meta['derived_path']).read_bytes()).hexdigest(),'status':'fail','reasons':[type(e).__name__+':'+str(e)],'scene':str(scene)}
                rec['house']=house;rec['color_thresholds']=cal['thresholds'];rec['color_calibration_path']=str(T/'color_calibration.json');records.append(rec);dump(T/(a.output_name+'.json'),records);print('NATIVE',house,meta['asset_id'],rec['status'],rec.get('reasons'),flush=True);progress(T,'原生 '+house+' '+meta['asset_id']+'：'+rec['status'],str(T/(a.output_name+'.json')),'汇总双场景指标、颜色及计数')
    dump(T/(a.output_name+'.json'),records)
if __name__=='__main__':main()
