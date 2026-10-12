"""Native surface/doorway and housing ROI algorithms copied from the admitted object task.

Source: v7_object_native.py at 6b68c056bb0d3817911a662a293b0a02c2e3458d.
No source checkout is an execution dependency. The caller supplies the runtime.
"""
import math
import numpy as np

def rotation(yaw):
    c,s=math.cos(yaw),math.sin(yaw);return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def ray(sim,r,p,d,length,exclude=None):
    hits=sim.cast_ray(r.habitat_sim.geo.Ray(r.magnum.Vector3(np.asarray(p,dtype=float)),r.magnum.Vector3(np.asarray(d,dtype=float))),length).hits
    hits=[h for h in hits if h.object_id!=exclude]
    return None if not hits else np.asarray(hits[0].point,dtype=float)

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
    center=np.asarray(anchor['center_m']);half=np.asarray(anchor.get('sizes_m',[0,0,0]))/2;delta=np.maximum(np.abs(np.asarray(point)-center)-half,0);return float(np.linalg.norm(delta))

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
    center=np.asarray(view['object_position'],float);points=[];anchors={'door':getattr(r,'semantic_anchors',{}).get('door',[]),'window':getattr(r,'semantic_anchors',{}).get('window',[])} if hasattr(r,'semantic_anchors') else semantic_anchors(sim);seeds=[]
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
        snap=np.asarray(getattr(r,"pathfinder",sim.pathfinder).snap_point(seed))
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
                p=floor+[0,height,0];hits=sim.cast_ray(r.habitat_sim.geo.Ray(r.magnum.Vector3(p),r.magnum.Vector3(d)),2.7).hits
                hit=None if not hits else np.asarray(hits[0].point,float)
                if hit is None or np.linalg.norm(hit-p)<.25:continue
                normal=np.asarray(hits[0].normal,float)
                if np.linalg.norm(normal)<1e-8 or abs(normal[1])>.20:continue
                normal[1]=0.;normal/=np.linalg.norm(normal)
                if normal@np.asarray(d)>0:normal=-normal
                base={'point':hit.tolist(),'floor_y':float(floor[1]),'normal':normal.tolist(),'native_normal_source':'actual stage ray-hit triangle normal, normalized horizontal yaw','surface':'wall','height_above_floor_m':height,'originating_native_floor_m':floor.tolist()}
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
