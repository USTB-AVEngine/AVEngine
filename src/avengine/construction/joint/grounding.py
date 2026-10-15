"""Solve root-height variants from real scan rays before native preparation."""
from __future__ import annotations
from copy import deepcopy
import math
import numpy as np
from .routes import ConstructionError
from .types import ActorPlan


def floor_contact_variants(context, actor, mesh, *, asset=None, yaw_rad=None, max_adjustment_m=.03):
    """Yield planned heights satisfying the original 15mm support-ray interval.

    This is proposal geometry, NOT native contact admission. Every variant must
    go through solve's common/family checks and strict native oracle. XY, heading
    and the supplied gait clock are unchanged; native execution may not lift it.
    """
    from avengine.capture.v7_visible_native import attachment_template
    from avengine.capture.v7_visible_attachment import rotation
    from .native import native_yaw
    ray=getattr(context,"floor_raycaster",None)
    if ray is None or getattr(ray,"source",None)!="visual_mesh":
        raise ConstructionError("height_refinement_requires_original_visual_floor")
    if not math.isfinite(max_adjustment_m) or not 0<max_adjustment_m<=.03:
        raise ValueError("bounded height refinement uses at most the v2 3cm support band")
    species=actor.body.species
    asset=asset or {"kind":"human"if species=="human"else("animal"if species in {"dog","cat"}else"object")}
    yaw=native_yaw(actor,asset) if yaw_rad is None else float(yaw_rad)
    _,local,template=attachment_template(np.asarray(mesh.vertices),"floor",asset.get("kind"),
                                         asset.get("landmarks_local_m"),mesh)
    root=np.asarray(actor.positions_m[0],float)
    world=np.asarray(local)@rotation(yaw).T+root
    ranges,_=ray.first_hit(world+[0,.1,0],np.tile([0.,-1.,0.],(len(world),1)))
    if not np.isfinite(ranges).all() or np.any(ranges>.22):
        raise ConstructionError("height_refinement_missing_original_support")
    deltas=ranges-.1
    lower=max(-max_adjustment_m,float(np.max(-.015-deltas)))
    upper=min(max_adjustment_m,float(np.min(.015-deltas)))
    if lower>upper:
        raise ConstructionError("height_refinement_no_common_support_interval",
                                {"deltas_m":deltas.tolist(),"interval_m":[lower,upper]})
    contact=-float(deltas.min())
    choices=[contact,contact+.001,contact+.002,contact-.001,0.]
    seen=set()
    for delta in choices:
        if not lower-1e-9<=delta<=upper+1e-9 or round(delta,10)in seen:
            continue
        seen.add(round(delta,10))
        positions=np.asarray(actor.positions_m,float).copy();positions[:,1]+=delta
        metadata=deepcopy(actor.metadata)
        metadata["floor_contact_refinement"]={"delta_y_m":delta,"original_root_m":root.tolist(),
            "support_probe_deltas_before_m":deltas.tolist(),"support_probe_deltas_predicted_m":(deltas+delta).tolist(),
            "allowed_translation_m":[lower,upper],"template":template,
            "authority":"original_visual_mesh_rays_proposal_only","native_contact_checks_required":True,
            "xy_or_distance_clock_changed":False}
        if metadata.get("fixed_clock_geometry"):
            metadata["fixed_clock_geometry"]["origin_m"][1]+=delta
        if metadata.get("waypoints_m"):
            for p in metadata["waypoints_m"]:p[1]+=delta
        yield ActorPlan(actor.entity_id,positions,actor.body,
                        None if actor.headings_rad is None else np.asarray(actor.headings_rad).copy(),metadata)
