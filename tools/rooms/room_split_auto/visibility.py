"""CPU ray visibility, finite-camera integer partitions, and honest optimality bounds."""
from __future__ import annotations
import math
import numpy as np
import shapely
from shapely.geometry import GeometryCollection
from scipy.spatial import cKDTree
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from tools.rooms.room_selection.navigation import ray_clear_batch,farthest_sample,cells_polygon
from tools.rooms.room_screening.geometry import union_projected_polygons


def visibility_matrix(mesh,points,camera_ids,p,max_distance):
    visible=np.zeros((len(camera_ids),len(points)),bool)
    distance=np.empty((len(camera_ids),len(points)))
    ends=points+[0,p['source_height_m'],0]
    for j,ci in enumerate(camera_ids):
        camera=points[ci]+[0,p['camera_height_m'],0]
        lengths=np.linalg.norm(ends-camera,axis=1);distance[j]=lengths
        ids=np.flatnonzero(lengths<=max_distance+1e-9)
        if len(ids):visible[j,ids]=ray_clear_batch(mesh,camera,ends[ids],p['ray_endpoint_tolerance_m'])
    return visible,distance


def grid_atoms(scope,nav_scope,points,step=0.25):
    """Exact floor intersections of world-grid cells, owned by the nearest nav sample.

    Areas come from continuous semantic polygons; the cells select partitions,
    not an alternative measurement definition. Furniture is never subtracted.
    """
    if not len(points):return [],np.array([]),np.array([])
    x0,z0,x1,z1=scope.bounds
    xs=np.arange(math.floor(x0/step)*step+step/2,x1+step,step)
    zs=np.arange(math.floor(z0/step)*step+step/2,z1+step,step)
    xx,zz=np.meshgrid(xs,zs,indexing='ij');xy=np.column_stack([xx.ravel(),zz.ravel()])
    cells=shapely.box(xy[:,0]-step/2,xy[:,1]-step/2,xy[:,0]+step/2,xy[:,1]+step/2)
    intersects=shapely.intersects(cells,scope);cells=cells[intersects];xy=xy[intersects]
    floor_cells=shapely.intersection(cells,scope)
    keep=shapely.area(floor_cells)>1e-12;floor_cells=floor_cells[keep];xy=xy[keep]
    owner=cKDTree(points[:,[0,2]]).query(xy)[1]
    atoms=[];floor_weights=[];nav_weights=[]
    for i in range(len(points)):
        atom=shapely.union_all(floor_cells[owner==i]) if np.any(owner==i) else GeometryCollection()
        atoms.append(atom);floor_weights.append(float(atom.area));nav_weights.append(float(atom.intersection(nav_scope).area))
    rebuilt=shapely.union_all(atoms)
    if rebuilt.symmetric_difference(scope).area>1e-7:raise ValueError('grid atoms do not cover the measured floor')
    return atoms,np.asarray(floor_weights),np.asarray(nav_weights)


def assess_visibility(mesh,points,nav_weights,p,coverage=0.8,max_distance=6.0,candidate_ids=None):
    if not len(points):return {'status':'not_run','reason':'NO_NAVMESH_GRID_POINTS','coverage_fraction':None,'camera_m':None,'farthest_distance_m':None,'meets_visibility':False}
    ids=farthest_sample(points,list(range(len(points))),min(128,len(points))) if candidate_ids is None else list(candidate_ids)
    vis,ds=visibility_matrix(mesh,points,ids,p,max_distance)
    weights=nav_weights if np.sum(nav_weights)>0 else np.ones(len(points))
    fractions=(vis@weights)/sum(weights)
    eligible=np.flatnonzero(ds.max(axis=1)<=max_distance+1e-7)
    chosen=int(eligible[np.argmax(fractions[eligible])]) if len(eligible) else int(np.argmax(fractions))
    return {'status':'measured','coverage_fraction':float(fractions[chosen]),'camera_m':(points[ids[chosen]]+[0,p['camera_height_m'],0]).tolist(),'farthest_distance_m':float(ds[chosen].max()),'meets_visibility':bool(len(eligible) and fractions[chosen]>=coverage-1e-8),'camera_candidates_tested':len(ids),'nav_grid_points':len(points),'nav_weight_area_m2':float(sum(weights)),'method':'0.25 m native-navmesh grid; intact raw GLB CPU first-hit rays; continuous clipped cell area weights','max_coverage_is_global':len(ids)==len(points)}


def partition_visibility(mesh,scope,nav_scope,points,p,coverage=0.8,max_distance=6.0,time_limit=20):
    if not len(points):return [scope],[],{'status':'unresolved','reason':'NO_NAVMESH_GRID_POINTS','minimum_piece_count_verified':False}
    atoms,fw,nw=grid_atoms(scope,nav_scope,points,p['grid_step_m'])
    camera_ids=farthest_sample(points,list(range(len(points))),min(64,len(points)))
    vis,dist=visibility_matrix(mesh,points,camera_ids,p,max_distance)
    # Every candidate is an actual 0.25 m-grid camera. Include an all-visible
    # centre when available; a lower bound from area can prove global minimum
    # even when a finite candidate set suffices for the construction.
    eligible=dist<=max_distance+1e-8
    ci,ui=np.nonzero(eligible);ncam=len(camera_ids);nedge=len(ci)
    if np.any(eligible.sum(axis=0)==0):
        return [scope],[],{'status':'unresolved','reason':'CAMERA_CANDIDATE_DISTANCE_COVER_INCOMPLETE','minimum_piece_count_verified':False}
    # y_j is camera/room activation, x_ij is ownership of exact floor atom i.
    rows=[];cols=[];vals=[];low=[];high=[]
    def constraint(columns,values,lo,hi):
        r=len(low);rows.extend([r]*len(columns));cols.extend(columns);vals.extend(values);low.append(lo);high.append(hi)
    for i in range(len(points)):
        e=np.flatnonzero(ui==i);constraint((ncam+e).tolist(),np.ones(len(e)).tolist(),1,1)
    for j in range(ncam):
        e=np.flatnonzero(ci==j)
        constraint([j,*list(ncam+e)],[-50,*fw[ui[e]].tolist()],-np.inf,0)
        constraint([j,*list(ncam+e)],[-6,*fw[ui[e]].tolist()],0,np.inf)
        constraint(list(ncam+e),(nw[ui[e]]*(vis[j,ui[e]].astype(float)-coverage)).tolist(),0,np.inf)
        for k in e:constraint([j,int(ncam+k)],[-1,1],-np.inf,0)
        owned=e[ui[e]==camera_ids[j]]
        if len(owned):constraint([j,int(ncam+owned[0])],[-1,1],0,np.inf)
    # Small spatial tie-break has total magnitude < 0.01, hence never trades
    # one extra room for compactness. It reduces disconnected assignments.
    objective=np.r_[np.ones(ncam),dist[ci,ui]/(max_distance*max(nedge,1)*100)]
    A=coo_matrix((vals,(rows,cols)),shape=(len(low),ncam+nedge)).tocsc()
    result=milp(objective,integrality=np.ones(ncam+nedge),bounds=Bounds(0,1),constraints=LinearConstraint(A,low,high),options={'time_limit':time_limit,'mip_rel_gap':0.00001,'presolve':True,'threads':1})
    receipt={'solver_status':int(result.status),'solver_message':result.message,'candidate_cameras':ncam,'nav_grid_points':len(points),'variables':ncam+nedge,'constraints':len(low),'area_lower_bound':int(math.ceil((scope.area-1e-8)/50)),'minimum_piece_count_verified':False,'candidate_set_optimal':result.status==0,'solver_mip_gap':float(result.mip_gap) if getattr(result,'mip_gap',None) is not None else None,'method':'integer assignment of exact floor-intersected 0.25 m cells; activation minimizes room count; distance and area-weighted visibility constrained; finite candidate cameras'}
    if result.x is None:return [scope],[],dict(receipt,status='unresolved',reason='VISIBILITY_INTEGER_PARTITION_NO_SOLUTION_WITHIN_CPU_BUDGET')
    # The room-count objective's tiny tie-break is below MILP feasibility
    # tolerances for large grids. Refine ownership with meaningful coefficients
    # after fixing the active cameras and room count, so compactness cannot buy
    # another room or weaken visibility/area/distance constraints.
    active=result.x[:ncam]>.5
    lower=np.zeros(ncam+nedge);upper=np.ones(ncam+nedge)
    lower[:ncam]=active.astype(float);upper[:ncam]=active.astype(float)
    compact_objective=np.r_[np.zeros(ncam),fw[ui]*dist[ci,ui]**2]
    compact=milp(compact_objective,integrality=np.ones(ncam+nedge),bounds=Bounds(lower,upper),constraints=LinearConstraint(A,low,high),options={'time_limit':time_limit,'mip_rel_gap':.00001,'presolve':True,'threads':1})
    receipt['compactness_refinement']={'solver_status':int(compact.status),'solver_message':compact.message,'active_cameras_fixed':int(active.sum()),'objective':'area-weighted squared camera distance; room count and camera selection fixed; all original constraints retained','candidate_camera_indices':np.flatnonzero(active).tolist()}
    if compact.x is None:return [scope],[],dict(receipt,status='unresolved',reason='COMPACT_ROOM_OWNERSHIP_NO_SOLUTION_WITHIN_CPU_BUDGET')
    result_x=compact.x
    assignment=np.full(len(points),-1,int)
    for edge in np.flatnonzero(result_x[ncam:]>0.5):assignment[ui[edge]]=ci[edge]
    if (assignment<0).any():return [scope],[],dict(receipt,status='unresolved',reason='VISIBILITY_INTEGER_ASSIGNMENT_INCOMPLETE')
    pieces=[];witnesses=[]
    for j in sorted(set(assignment)):
        ids=np.flatnonzero(assignment==j);piece=shapely.union_all([atoms[i] for i in ids])
        if piece.is_empty:continue
        cov=float(sum(nw[ids]*vis[j,ids])/sum(nw[ids])) if sum(nw[ids]) else 0.0
        far=float(dist[j,ids].max())
        if piece.area>50+1e-6 or piece.area<6-1e-6 or cov<coverage-1e-6 or far>max_distance+1e-6:
            return [scope],[],dict(receipt,status='unresolved',reason='VISIBILITY_INTEGER_RESULT_VALIDATION_FAILED')
        pieces.append(piece);witnesses.append({'status':'measured','camera_m':(points[camera_ids[j]]+[0,p['camera_height_m'],0]).tolist(),'coverage_fraction':cov,'farthest_distance_m':far,'meets_visibility':True,'nav_grid_points':len(ids),'nav_weight_area_m2':float(sum(nw[ids])),'method':'integer-assigned native navmesh grid; intact raw scan rays'})
    if shapely.union_all(pieces).symmetric_difference(scope).area>1e-7:raise ValueError('visibility partition lost floor area')
    receipt.update(status='partitioned',piece_count=len(pieces),minimum_piece_count_verified=len(pieces)==receipt['area_lower_bound'],minimum_proof=('achieves universal area lower bound ceil(A/50)' if len(pieces)==receipt['area_lower_bound'] else 'global minimum unverified; optimality applies only to tested camera set' if result.status==0 else 'CPU time limit; feasible partition only'))
    return pieces,witnesses,receipt
