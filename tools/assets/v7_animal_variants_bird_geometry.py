"""Separate measured wood from generated bird, scale beak-tail, and author a silent stand."""
import math
import numpy as np
import trimesh
from scipy.ndimage import label
from scipy.spatial.transform import Rotation
from v7_animal_variants_color import rgb_to_lab

def vertex_rgb(mesh):
    visual=mesh.visual;uv=np.asarray(visual.uv);image=getattr(visual.material,'baseColorTexture',None)
    if image is None:image=getattr(visual.material,'image',None)
    if image is None:raise RuntimeError('bird texture unavailable for wood/foot masks')
    image=np.array(image.convert('RGB'));x=np.rint(np.clip(uv[:,0],0,1)*(image.shape[1]-1)).astype(int);y=np.rint((1-np.clip(uv[:,1],0,1))*(image.shape[0]-1)).astype(int);return image[y,x]

def two_feet(v,rgb,rod_top=None):
    lab=rgb_to_lab(rgb);h=np.mod(np.degrees(np.arctan2(lab[:,2],lab[:,1])),360);c=np.hypot(lab[:,1],lab[:,2]);span=np.ptp(v[:,1]);lo=v[:,1].min();center=np.median(v[:,[0,2]],axis=0);horizontal=np.ptp(v[:,[0,2]],axis=0);in_body=(np.abs(v[:,0]-center[0])<horizontal[0]*.45)&(np.abs(v[:,2]-center[1])<horizontal[1]*.45);skin=((h<35)|(h>330))&(c>5)&(lab[:,0]<85)&(v[:,1]<lo+.60*span)&in_body
    if rod_top is not None:skin &= np.abs(v[:,1]-rod_top)<.12*span
    candidates=v[skin]
    if len(candidates)<30:
        # Geometric fallback is explicitly qualified by two isolated low clusters.
        bound=lo+.25*span if rod_top is None else rod_top+.06*span;candidates=v[(v[:,1]<=bound)&in_body]
    if len(candidates)<30:raise RuntimeError('bird two-foot candidate region missing')
    step=span*.014;xy=candidates[:,[0,2]];base=xy.min(0);ij=np.floor((xy-base)/step).astype(int);grid=np.zeros(tuple(ij.max(0)+3),bool);grid[ij[:,0]+1,ij[:,1]+1]=True;labs,n=label(grid,np.ones((3,3)));groups=[]
    for i in range(1,n+1):
        pts=candidates[labs[ij[:,0]+1,ij[:,1]+1]==i]
        if len(pts)>=max(12,.025*len(candidates)):groups.append(pts)
    groups.sort(key=len,reverse=True)
    if not groups:raise RuntimeError('bird foot-contact patch missing')
    points=[]
    for pts in groups[:2]:
        minimum=pts[:,1].min();center=pts.mean(0);near=np.linalg.norm(v[:,[0,2]]-center[[0,2]],axis=1)<span*.065;gray_skin=(c<12)&(lab[:,0]>20)&(lab[:,0]<70);region=near&((h<35)|(h>330)|gray_skin)&(v[:,1]>minimum-.09*span)&(v[:,1]<minimum+.07*span);refined=v[region];refined=refined if len(refined)>20 else pts;minimum=refined[:,1].min();bottom=refined[refined[:,1]<minimum+.012*span];points.append(bottom.mean(0))
    points=np.array(points)
    if len(points)>1 and np.linalg.norm(points[0]-points[1])<.045*span:points=points[:1]
    return points,{'candidate_vertices':len(candidates),'cluster_sizes':[len(x) for x in groups],'measured_contact_regions':len(points),'anatomical_two_feet_count':'not_inferred_from_texture_clusters','skin_color_method':'actual low skin-texture contact regions; a connected toe mask is a contact patch, not proof of one anatomical foot'}

def normalize(mesh,item):
    rgb=vertex_rgb(mesh);v=mesh.vertices.copy();span=np.ptp(v[:,1]);initial_feet,initial_foot_details=two_feet(v,rgb)
    lab=rgb_to_lab(rgb);c=np.hypot(lab[:,1],lab[:,2]);h=np.mod(np.degrees(np.arctan2(lab[:,2],lab[:,1])),360)
    line=initial_feet[1]-initial_feet[0] if len(initial_feet)>1 else np.array([1.,0,0]);line/=np.linalg.norm(line);center=initial_feet.mean(0)+np.array([0,.035*span,0]);q=v-center;along=q@line;distance=np.linalg.norm(q-along[:,None]*line,axis=1)
    rod_region=(distance<.070*span)&(np.abs(along)<span*.65)
    wood_color=((h>=40)&(h<=110)&(c>=8)&(c<=60)&(lab[:,0]<88))|((c<12)&(lab[:,0]<45))
    protected=np.min(np.stack([np.linalg.norm(v[:,[0,2]]-point[[0,2]],axis=1) for point in initial_feet]),axis=0)<span*.050
    from v7_animal_variants_rod import rod_faces
    consensus,rod_diagnostic=rod_faces(v,mesh.faces,rgb,initial_feet)
    rod_vertex=rod_region&wood_color&~protected;face_wood=(rod_vertex[mesh.faces].sum(1)>=2)|consensus;removed=int(face_wood.sum());rod_top=None
    if removed>50:
        rod_points=v[np.unique(mesh.faces[face_wood])];extent=np.ptp(rod_points,axis=0)
        if max(extent[0],extent[2])<span*.06:raise RuntimeError('generated rod removal region too small to qualify')
        mesh.update_faces(~face_wood);mesh.remove_unreferenced_vertices();v=mesh.vertices.copy();rgb=vertex_rgb(mesh)
    # Determine heading on the bird, after generated wood is removed.
    xz=v[:,[0,2]];_,_,vh=np.linalg.svd((xz-xz.mean(0))[::max(1,len(xz)//10000)],full_matrices=False);axis=vh[0];head_upper=v[:,1]>=np.quantile(v[:,1],.72);sign=np.sign(np.median((xz[head_upper]-xz.mean(0))@axis)) or 1;front=np.array([axis[0],0,axis[1]])*sign;theta=math.pi-math.atan2(front[0],front[2]);rot=Rotation.from_rotvec([0,theta,0]).as_matrix();v=v@rot.T;points,foot_details=two_feet(v,rgb,rod_top)
    angle=0.;level=np.eye(3)
    if len(points)>1:
        line=points[1]-points[0];target=line.copy();target[1]=0;target*=np.linalg.norm(line)/max(np.linalg.norm(target),1e-12);a=line/np.linalg.norm(line);b=target/np.linalg.norm(target);axis=np.cross(a,b);angle=math.acos(np.clip(a@b,-1,1))
        if math.degrees(angle)<=25:level=Rotation.from_rotvec(axis/max(np.linalg.norm(axis),1e-12)*angle).as_matrix()
        else:
            # Do not roll a bird merely because color patches occur at different heights.
            points=points[np.argsort(points[:,1])[:1]];angle=0.;foot_details['level_fit']='texture regions at different heights; retain native bird posture and qualify lowest contact patch'
    v=v@level.T;points=points@level.T;top=v[v[:,1]>=np.quantile(v[:,1],.65)];head=top[top[:,2]<=np.quantile(top[:,2],.35)];beak=head[head[:,2]<=np.quantile(head[:,2],.025)].mean(0);rear=v[v[:,1]<=np.quantile(v[:,1],.55)];tail=rear[rear[:,2]>=np.quantile(rear[:,2],.985)].mean(0);raw_length=float(np.linalg.norm(beak-tail));scale=item['physical_size']['target_m']/raw_length;perch_top=.35;origin=points.mean(0);translation=np.array([0,perch_top,0]);v=(v-origin)*scale+translation;points=(points-origin)*scale+translation;beak=(beak-origin)*scale+translation;tail=(tail-origin)*scale+translation;head=(head-origin)*scale+translation;mesh.vertices=v
    # The rail axis follows the measured foot-to-foot horizontal line.
    foot_line=points[1]-points[0] if len(points)>1 else np.array([.05,0,0]);rail_yaw=math.atan2(foot_line[2],foot_line[0]);rod_rotation=Rotation.from_euler('y',-rail_yaw).as_matrix()@Rotation.from_rotvec([0,math.pi/2,0]).as_matrix();rail_radius=.007;rail_length=max(.12,float(np.linalg.norm(foot_line))+.06);wood_material=trimesh.visual.material.PBRMaterial(name='OwnedTaskMatteWood',baseColorFactor=[130,85,48,255],metallicFactor=0,roughnessFactor=1);scene=trimesh.Scene();scene.add_geometry(mesh,geom_name='GeneratedAnimal');stand=trimesh.Scene()
    for name,radius,length,center,rotation in [('PerchRoundBase',.08,.012,[0,.006,0],Rotation.from_rotvec([math.pi/2,0,0]).as_matrix()),('PerchPost',.009,perch_top-.016,[0,(perch_top-.016)/2+.012,0],Rotation.from_rotvec([math.pi/2,0,0]).as_matrix()),('PerchRod',rail_radius,rail_length,[0,perch_top-rail_radius,0],rod_rotation)]:
        part=trimesh.creation.cylinder(radius=radius,height=length,sections=32);part.vertices=part.vertices@rotation.T+center;part.visual=trimesh.visual.texture.TextureVisuals(material=wood_material);scene.add_geometry(part,geom_name=name);stand.add_geometry(part.copy(),geom_name=name)
    marks={'mouth':beak.tolist(),'head_center':head.mean(0).tolist(),'head_bbox_min':head.min(0).tolist(),'head_bbox_max':head.max(0).tolist(),'tail_tip':tail.tolist(),**{'perch_contact_'+str(i):point.tolist() for i,point in enumerate(points)}};detail={'scale':scale,'beak_to_tail_length_m':float(np.linalg.norm(beak-tail)),'size_measurement_method':'straight measured upper-front beak tip to low rear tail tip on generated mesh','raw_beak_to_tail_length':raw_length,'generated_wood_triangles_removed':removed,'wood_removal_method':'measured two-foot axis cylinder; actual brown or dark neutral texture; foot neighborhoods protected; no low-height slicing','initial_foot_clusters':initial_foot_details,'transverse_rod_diagnostic':rod_diagnostic,'body_max_extent_to_length_ratio':float(np.max(np.ptp(v,axis=0))/item['physical_size']['target_m']),'yaw_rad':theta,'feet_level_rotation_deg':math.degrees(angle),'head_above_contact_m':float(head[:,1].mean()-perch_top),'minimum_bird_y_m':float(v[:,1].min()),'foot_height_difference_m':float(np.ptp(points[:,1])),'contact_regions_separation_m':float(np.linalg.norm(points[0]-points[1])) if len(points)>1 else None,'bird_stand_scope':'flat stand base support and measured skin-contact patch; no anatomical foot count claimed',**foot_details};perch={'base_radius_m':.08,'base_height_m':.012,'perch_top_m':perch_top,'rail_radius_m':rail_radius,'rail_length_m':rail_length,'structure':'generated bird retained; measured textured wood faces removed; task-authored silent horizontal rail, post, flat base','generated_wood_triangles_removed':removed}
    if removed>30000 or np.ptp(v[:,1])<.035 or detail['foot_height_difference_m']>.002 or detail['head_above_contact_m']<.025 or detail['minimum_bird_y_m']<.02:raise RuntimeError('bird pose or size gate')
    return scene,mesh,stand,v,marks,detail,perch
