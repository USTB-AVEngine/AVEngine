"""Exact straight critical windows connected by smooth native NavMesh turns."""
import math
from types import SimpleNamespace
import numpy as np
from shapely.geometry import shape,Point,LineString,box

def stadium_samples(distance,length,radius):
    s=np.mod(np.asarray(distance,float),2*length+2*math.pi*radius);p=np.zeros((len(s),2));v=p.copy();ids=np.full(len(s),-1)
    one=s<length;p[one]=np.column_stack((-length/2+s[one],np.full(one.sum(),radius)));v[one]=[1,0];ids[one]=0
    two=(s>=length)&(s<length+math.pi*radius);a=(s[two]-length)/radius;p[two]=np.column_stack((length/2+radius*np.sin(a),radius*np.cos(a)));v[two]=np.column_stack((np.cos(a),-np.sin(a)))
    three=(s>=length+math.pi*radius)&(s<2*length+math.pi*radius);q=s[three]-length-math.pi*radius;p[three]=np.column_stack((length/2-q,np.full(three.sum(),-radius)));v[three]=[-1,0];ids[three]=1
    four=s>=2*length+math.pi*radius;a=(s[four]-2*length-math.pi*radius)/radius;p[four]=np.column_stack((-length/2-radius*np.sin(a),-radius*np.cos(a)));v[four]=np.column_stack((-np.cos(a),np.sin(a)))
    return p,v,ids

def straight_window_routes(video,entry,scene,distance,local_forward,request):
    times=np.arange(len(distance))/30.;windows=request['straight_motion_windows_s'];sound_windows=request.get('sounding_straight_windows_s',[])
    slices=[np.flatnonzero((times>=float(lo))&(times<=float(hi)))for lo,hi in windows]
    poly=shape(scene['production_scenario']['motion_floor_polygon']);margin=.32 if entry['declaration']['kind']=='human'else .25;free=(poly.convex_hull if request.get('motion_quartet_route') else poly).buffer(-margin)
    x0,z0,x1,z1=free.bounds;centres=[free.representative_point()]+[Point(x,z)for x in np.linspace(x0,x1,15)for z in np.linspace(z0,z1,15)];centres=[p for p in centres if free.covers(p)]
    start=np.asarray(entry['applied_root_input'][:3,3]);centres.sort(key=lambda p:(p.x-start[0])**2+(p.y-start[2])**2)
    obstacles=[]
    for actor in video.actors:
        if actor is entry or actor.get('phase2_layout_qualified')is False:continue
        root=actor['applied_root_input'];v=np.asarray(actor['original_mesh'].vertices)@root[:3,:3].T+root[:3,3];obstacles.append(box(v[:,0].min()-margin,v[:,2].min()-margin,v[:,0].max()+margin,v[:,2].max()+margin))
    listener=np.asarray(scene['listener_m']);theta=math.radians(scene['camera_yaw_knots'][0]['yaw_deg']);camera=np.array([math.sin(theta),0.,-math.cos(theta)]);right=np.cross(camera,[0.,1.,0.]);mouth=np.asarray(entry['declaration'].get('emitter_local_m')or[0.,1.,0.]);base=math.atan2(local_forward[2],local_forward[0]);proposal=0
    required=max(float(distance[ids[-1]]-distance[ids[0]])for ids in slices if len(ids))
    template=request.get('motion_quartet_route')
    lengths=[template['straight_length_m']] if template else [x for x in (1.8,2.1,2.4,2.7,3.,3.3,3.6,4.,4.5)if x>=required+.03]
    for length in lengths:
      for radius in ([template['turn_radius_m']] if template else (.75,.65,.55,.85,.45,1.)):
        period=2*length+2*math.pi*radius;patterns=[]
        for offset in ([template.get('phase_distance_offset_m',0.)] if template else np.linspace(0,period,192,endpoint=False)):
          if template and template.get('single_pass'):
            p=np.column_stack((-length/2+np.asarray(distance),np.zeros(len(distance))));tangent=np.tile([1.,0.],(len(distance),1));segments=np.zeros(len(distance),int)
          else:p,tangent,segments=stadium_samples(np.asarray(distance)+offset,length,radius)
          if any(not len(ids)or segments[ids[0]]<0 or np.any(segments[ids]!=segments[ids[0]])for ids in slices):continue
          patterns.append((offset,p,tangent))
        if not patterns:continue
        headings=[template['heading_rad']+delta for delta in (0.,math.pi/16,-math.pi/16,math.pi/8,-math.pi/8)] if template else np.arange(16)*math.pi/8
        for heading in headings:
          rotation=np.array([[math.cos(heading),-math.sin(heading)],[math.sin(heading),math.cos(heading)]])
          selected_centres=centres
          if template:
            proposed=Point(*template['centre_xz_m'])
            selected_centres=[proposed]+sorted((p for p in centres if p.distance(proposed)>1e-8),key=lambda p:p.distance(proposed))
          for centre in selected_centres:
            for offset,p,tangent in patterns:
              xz=p@rotation.T+[centre.x,centre.y];line=LineString(xz)
              if not free.covers(line)or any(line.intersects(o)for o in obstacles):continue
              positions=np.column_stack((xz[:,0],np.full(len(xz),start[1]),xz[:,1]));t=tangent@rotation.T;yaws=np.unwrap(base-np.arctan2(t[:,1],t[:,0]))
              if np.min(np.linalg.norm(positions[:,[0,2]]-listener[[0,2]],axis=1))<.6:continue
              if template and template.get('distance_branches'):
                c,s=np.cos(yaws),np.sin(yaws)
                emit=positions+np.column_stack((c*mouth[0]+s*mouth[2],np.full(len(times),mouth[1]),-s*mouth[0]+c*mouth[2]))
                radial=np.linalg.norm(emit-listener,axis=1);bad=False
                for branch in template['distance_branches']:
                  epoch=float(branch['offset_s'])
                  relevant=[(lo,hi)for lo,hi in sound_windows if epoch<=lo<epoch+10]
                  lo,hi=relevant[0]if relevant else(epoch+.3,epoch+2.7)
                  values=radial[(times>=lo)&(times<=hi)];sign=-1 if branch['branch']=='nearer'else 1
                  if len(values)<2 or sign*(values[-1]-values[0])<request.get('minimum_distance_change_m',.4)or np.min(sign*np.diff(values))<-.005:bad=True;break
                if bad:continue
              if request.get('distance_trend'):
                radial=np.linalg.norm(positions-listener,axis=1);sign=-1 if request['distance_trend']=='approaching' else 1;bad=False
                for lo,hi in request.get('distance_trend_intervals_s',[]):
                  values=radial[(times>=lo)&(times<=hi)]
                  if len(values)<2 or sign*(values[-1]-values[0])<request.get('minimum_distance_change_m',1.5)or np.min(sign*np.diff(values))<-.001:bad=True;break
                if bad:continue
              visible=True
              for lo,hi in sound_windows:
                for at in (lo,(lo+hi)/2,hi):
                  i=min(len(times)-1,round(at*30));c,s=math.cos(yaws[i]),math.sin(yaws[i]);rot=np.array([[c,0,s],[0,1,0],[-s,0,c]]);delta=positions[i]+rot@mouth-listener;depth=float(delta@camera)
                  if depth<=0 or abs(math.degrees(math.atan2(delta@right,depth)))>37 or abs(math.degrees(math.atan2(delta[1],depth)))>24:visible=False;break
                if not visible:break
              if not visible:continue
              snaps=np.asarray([video.pathfinder.snap_point(p)for p in positions]);errors=np.linalg.norm((snaps-positions)[:,[0,2]],axis=1)
              if not np.isfinite(snaps).all()or errors.max()>.002:continue
              positions[:,[0,2]]=snaps[:,[0,2]];links=[]
              for i,j in zip(np.linspace(0,len(snaps)-1,9,dtype=int)[:-1],np.linspace(0,len(snaps)-1,9,dtype=int)[1:]):
                path=video.hs.ShortestPath();path.requested_start=snaps[i];path.requested_end=snaps[j]
                if not video.pathfinder.find_path(path):break
                links.append({'start_frame':int(i),'end_frame':int(j),'native_geodesic_m':float(path.geodesic_distance)})
              if len(links)!=8:continue
              proposal+=1
              yield SimpleNamespace(positions=positions,yaws=yaws,receipt={'revision':'v3-timed-straight-windows-native-stadium-v1','proposal':proposal,'route_kind':'straight critical windows with smooth turns only outside them','straight_windows_s':windows,'straight_length_m':length,'turn_radius_m':radius,'phase_distance_offset_m':float(offset),'centre_xz_m':[centre.x,centre.y],'heading_rad':float(heading),'proposal_centre_xz_m':None if not template else template['centre_xz_m'],'automatic_native_retry_preserves_camera':bool(template),'root_samples':len(times),'maximum_horizontal_snap_error_m':float(errors.max()),'native_waypoint_connections':links,'heading_follows_route_tangent':True,'native_full_pose_checks_still_required':True})
