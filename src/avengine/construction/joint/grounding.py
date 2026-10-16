"""Solve root-height variants from real scan rays before native preparation."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from .routes import ConstructionError
from .types import ActorPlan


def floor_contact_variants(context, actor, mesh, *, asset=None, yaw_rad=None, max_adjustment_m=.03):
    """Construct real scan support, then retain imperfect foot contact as evidence.

    The legacy max_adjustment_m argument remains API-compatible but is no longer
    a body/floor admission threshold. Void, another floor and furniture support
    remain illegal; original native non-floor checks still follow.
    """
    from avengine.capture.v7_visible_native import attachment_template
    from avengine.capture.v7_visible_attachment import rotation
    from .native import native_yaw
    ray=getattr(context,"floor_raycaster",None)
    if ray is None or getattr(ray,"source",None)!="visual_mesh":
        raise ConstructionError("height_refinement_requires_original_visual_floor")
    species=actor.body.species
    asset=asset or {"kind":"human"if species=="human"else("animal"if species in {"dog","cat"}else"object")}
    yaw=native_yaw(actor,asset) if yaw_rad is None else float(yaw_rad)
    _,local,template=attachment_template(np.asarray(mesh.vertices),"floor",asset.get("kind"),
                                         asset.get("landmarks_local_m"),mesh)
    root=np.asarray(actor.positions_m[0],float)
    world=np.asarray(local)@rotation(yaw).T+root
    from avengine.construction.contact_policy import contact_quality_observation, FLOOR_STEP_RANGE_M
    # Older leaf adapters expose only the original scan raycaster. Their
    # supplied root is the reference proposal; real RoomContext uses its room
    # floor authority. No extra adapter-only admission requirement is added.
    reference=float(getattr(context,"room",{}).get("measured_floor_y_m",root[1]))
    roots=np.asarray(actor.positions_m,float)
    root_origins=roots.copy();root_origins[:,1]=reference+.2
    root_ranges,_=ray.first_hit(root_origins,np.tile([0.,-1.,0.],(len(roots),1)))
    floor_y=root_origins[:,1]-root_ranges
    if (not np.isfinite(root_ranges).all() or np.any(root_ranges>.85)
            or np.any(np.abs(floor_y-reference)>FLOOR_STEP_RANGE_M)):
        raise ConstructionError("height_refinement_missing_original_support")
    origins=world.copy();origins[:,1]=reference+.2
    ranges,_=ray.first_hit(origins,np.tile([0.,-1.,0.],(len(world),1)))
    valid=np.isfinite(ranges)&(ranges<=.85)
    support_y=origins[:,1]-ranges
    if not valid.any() or np.any(np.abs(support_y[valid]-reference)>FLOOR_STEP_RANGE_M):
        raise ConstructionError("height_refinement_missing_original_support")
    # Match the lowest real attachment point to the measured root floor. The
    # other feet may have residuals or missing rays; neither vetoes construction.
    delta=float(floor_y[0]-world[:,1].min())
    positions=roots.copy()
    positions[:,1]=floor_y-(world[:,1].min()-root[1])
    deltas=world[valid,1]-support_y[valid]
    metadata=deepcopy(actor.metadata)
    observation=contact_quality_observation(deltas=deltas+delta,
        root_support_verified=True,root_translation_y_m=delta,
        missing_attachment_probe_indices=np.flatnonzero(~valid).tolist(),
        requested_attachment_probe_count=len(world),measured_attachment_probe_count=int(valid.sum()),
        root_floor_y_range_m=[float(floor_y.min()),float(floor_y.max())])
    metadata["contact_quality_observation"]=observation
    metadata["floor_contact_refinement"]={"delta_y_m":delta,"original_root_m":root.tolist(),
        "support_probe_deltas_before_m":deltas.tolist(),"support_probe_deltas_predicted_m":(deltas+delta).tolist(),
        "template":template,"authority":"original_visual_mesh_rays_proposal_only",
        "native_contact_checks_required":True,"xy_or_distance_clock_changed":False,
        "contact_quality_observation":observation}
    if metadata.get("fixed_clock_geometry"):
        metadata["fixed_clock_geometry"]["origin_m"][1]+=delta
    if metadata.get("waypoints_m"):
        for point in metadata["waypoints_m"]:point[1]+=delta
    yield ActorPlan(actor.entity_id,positions,actor.body,
                    None if actor.headings_rad is None else np.asarray(actor.headings_rad).copy(),metadata)
