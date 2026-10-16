"""Two independently prepared visual worlds for one registered four-member parent."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import math
import numpy as np
from .types import ActorPlan, Certificate, CheckResult, PreparedCandidate, ScenePlan
from .certification import certify_plan
from .routes import ConstructionError
from .solver import _awaits_only_native_sweep, _resolve_native_sweep_pending


def _close_candidates(candidates):
    seen=set();errors=[]
    # Habitat simulators created later can borrow the first simulator's GL
    # context. Release in reverse acquisition order so that owner outlives
    # every dependent world, including failure/exception cleanup.
    for candidate in reversed(list(candidates)):
        token=id(candidate.payload)if candidate.payload is not None else id(candidate)
        if token in seen:
            continue
        seen.add(token)
        closer=candidate.close or getattr(candidate.payload,"close",None)
        if closer is not None:
            try: closer()
            except Exception as exc: errors.append(exc)
    if errors:
        raise RuntimeError("native_bundle_close_failed")from errors[0]


class PreparedNativeBundle:
    """Own two live payloads; row is exactly the v0 baseline row."""
    def __init__(self, owners, certificate, directory=None):
        self._owners=dict(owners)
        self.visuals={key:value.payload for key,value in owners.items()}
        self.row=self.visuals["v0"].row
        self.certificate=certificate
        self.directory=Path(directory)if directory is not None else None
        self.closed=False

    def close(self):
        if not self.closed:
            self.closed=True
            _close_candidates(self._owners.values())

    def bind_solution(self, record):
        from avengine.timeline.continuous_pilot import write_json
        group={**deepcopy(record),"prepared_visual_variants":["v0","v1"],
               "prepared_visual_count":2,"same_prepared_visuals_required":True}
        for name,payload in self.visuals.items():
            payload.bind_solution({**group,"prepared_visual_variant":name})
        # Keep the baseline row object held by the native consumer.
        self.row["joint_solution"]=deepcopy(group)
        if self.directory is not None:
            write_json(self.directory/"joint_solution.json",group)

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()


def swapped_visual_plan(plan, swap_asset_ids):
    """Swap physical endpoint slots, retaining each asset's body and event binding."""
    left,right=tuple(swap_asset_ids)
    if left==right or not {left,right}<=set(plan.actors):
        raise ConstructionError("invalid_registered_visual_swap")
    mapping={left:right,right:left}
    actors=deepcopy(plan.actors)
    for key,slot in mapping.items():
        actor=actors[key];source=plan.actors[slot]
        actor.positions_m=np.asarray(source.positions_m,float).copy()
        actor.headings_rad=None if source.headings_rad is None else np.asarray(source.headings_rad,float).copy()
        for name in ("validation","floor_contact_refinement","framing_constraint","native_yaw_rad","native_yaw_rad_per_frame"):
            actor.metadata.pop(name,None)
        if "native_yaw_rad"in source.metadata:
            actor.metadata["native_yaw_rad"]=source.metadata["native_yaw_rad"]
        actor.metadata["registered_visual_slot_from_asset"]=slot
    metadata=deepcopy(plan.metadata)
    metadata.pop("joint_solution",None)
    if "roles"in metadata:
        metadata["roles"]={mapping.get(key,key):value for key,value in metadata["roles"].items()}
    metadata["registered_visual_swap"]=[left,right]
    sounds=tuple(replace(event,entity_id=mapping.get(event.entity_id,event.entity_id))for event in plan.sounds)
    return ScenePlan(plan.camera,actors,sounds,plan.frame_rate_hz,plan.duration_s,metadata)


def prepare_native_bundle(context, request, plans, prepare_member, *, group_check=None, directory=None):
    """One solve callback, with per-world geometry/native proof and atomic ownership.

    prepare_member(visual, context, member_request, plan) returns PreparedCandidate.
    Its request defers family obligations to group_check; all original common,
    physical and acoustic checks must still be run by the native preparer.
    """
    if set(plans)!={"v0","v1"}or any(not isinstance(p,ScenePlan)for p in plans.values()):
        raise ValueError("exactly v0 and v1 ScenePlans required")
    base=plans["v0"];checks=[];geometry={}
    for key,plan in plans.items():
        aligned=(plan.camera==base.camera and plan.frame_rate_hz==base.frame_rate_hz
                 and plan.duration_s==base.duration_s and set(plan.actors)==set(base.actors))
        static=all(np.asarray(a.positions_m).ndim==2 and not np.any(np.diff(a.positions_m,axis=0))
                   and(a.headings_rad is None or not np.any(np.diff(a.headings_rad)))
                   and(a.metadata.get("native_yaw_rad_per_frame")is None
                       or not np.any(np.diff(a.metadata["native_yaw_rad_per_frame"])))
                   for a in plan.actors.values())
        checks.append(CheckResult(key+":common_world_contract","pass"if aligned and static else"fail",
                                  "same_camera_clock_entities_and_static_poses"))
        geometry[key]=certify_plan(context,plan).checks
    def tagged(key,values):
        return tuple(replace(c,name=key+":"+c.name,evidence={**c.evidence,"visual_variant":key})for c in values)
    if (any(c.status!="pass"for c in checks)
            or any(not _awaits_only_native_sweep(values)for values in geometry.values())):
        return PreparedCandidate(None,tuple(checks)+tuple(c for k,v in geometry.items()for c in tagged(k,v)))
    owned={};bundle=None
    def cleanup():
        if bundle is not None:bundle.close()
        else:_close_candidates(owned.values())
    member_request=replace(request,options={**request.options,"require_native_preflight":False})
    try:
        for key in ("v0","v1"):
            candidate=prepare_member(key,context,member_request,plans[key])
            if not isinstance(candidate,PreparedCandidate):
                raise TypeError("bundle member preparer must return PreparedCandidate")
            owned[key]=candidate
            native=tuple(candidate.checks)or(CheckResult("native:member_checks","unknown","missing_member_checks"),)
            resolved=_resolve_native_sweep_pending(geometry[key],native)
            checks.extend(tagged(key,resolved+native))
            # Only v0 aliases may resolve the outer solve's v0 coarse-map pending.
            # v1 geometry above is resolved exclusively by v1's own native evidence.
            if key=="v0":checks.extend(native)
            payload=candidate.payload
            fields=("video","scene","row","house","assets","program","directory","certificate","closed","close","bind_solution")
            live=(payload is not None and all(hasattr(payload,name)for name in fields)
                  and not payload.closed and payload.video is not None
                  and getattr(payload.video,"sim",None)is not None and payload.certificate.passed)
            checks.append(CheckResult(key+":native:live_payload","pass"if live else"unknown",
                                      "same_live_prepared_member_required"))
            if any(c.status!="pass"for c in checks):
                cleanup()
                return PreparedCandidate(None,tuple(checks))
        a,b=owned["v0"].payload,owned["v1"].payload
        independent=(a is not b and a.video is not b.video and a.video.sim is not b.video.sim
                     and a.scene is not b.scene and a.row is not b.row)
        checks.append(CheckResult("native:two_visual_worlds","pass"if independent else"fail",
                                  "distinct_live_worlds_and_scene_records"))
        parent_contract=(bool(a.scene.get("scene_id"))and
            a.scene.get("scene_id")==b.scene.get("scene_id")==a.row.get("scene_id")==b.row.get("scene_id")
            and a.house==b.house and a.program==b.program)
        checks.append(CheckResult("native:registered_parent_contract","pass"if parent_contract else"fail",
                                  "same_scene_id_house_and_original_program"))
        if not independent or not parent_contract:
            cleanup()
            return PreparedCandidate(None,tuple(checks))
        bundle=PreparedNativeBundle(owned,Certificate(tuple(checks),"two_native_prepared_worlds"),directory)
        extra=tuple(group_check(bundle))if group_check is not None else ()
        if not extra:
            extra=(CheckResult("native:registered_group_obligations","unknown","group_native_family_checks_not_supplied"),)
        checks.extend(extra)
        certificate=Certificate(tuple(checks),"planning_and_two_native_visual_preflights")
        bundle.certificate=certificate
        if directory is not None:
            from avengine.timeline.continuous_pilot import write_json
            write_json(Path(directory)/"native_bundle_certificate.json",asdict(certificate))
        if not certificate.passed:
            cleanup()
            return PreparedCandidate(None,certificate.checks)
        return PreparedCandidate(bundle,certificate.checks,bundle.close)
    except Exception:
        cleanup()
        raise


def prepare_registered_bundle(context, request, plan, *, row_factory, house, catalog, records,
        semantic, runtime, output_dir, visual_plans=None, family_checks=None, group_check=None,
        roles_by_visual=None, minimum_visible_pixels_by_visual=None, appearance_ids=()):
    """Prepare the original registered v0/v1 visuals for the four AV members.

    The registered sampler compiles the baseline program once. v1 swaps the
    first two assets between their original endpoints, as in the original
    registered producer. No body is dropped and no sound/time is resampled.
    Caller-supplied visual_plans may refine Y/heading before native preparation;
    both views must retain the intended endpoint XZ slots and event bindings.
    """
    import tempfile
    from .native import prepare_native, native_yaw
    from avengine.capture.v7_registered_native import NativeRegisteredVideo
    from avengine.timeline.v7_registered_sampling import choose_assets, build_registered_scene
    from avengine.timeline.v7_visible_sources import production_audio_record_views
    from avengine.capture.v7_visible_attachment import rotation
    from avengine.timeline.continuous_pilot import write_json
    row=deepcopy(row_factory(plan))
    if "row"in row and "production_scenario"not in row:row=row["row"]
    assets=choose_assets(catalog,row)
    ids=[a["asset_id"]for a in assets]
    if len(ids)<2 or len(set(ids))!=len(ids)or set(ids)!=set(plan.actors):
        raise ConstructionError("registered_plan_must_include_every_selected_asset")
    if plan.duration_s!=30. or plan.frame_rate_hz!=30.:
        raise ConstructionError("registered_bundle_requires_original_30s_30Hz_plan")
    expected=swapped_visual_plan(plan,ids[:2])
    plans={"v0":plan,"v1":expected}if visual_plans is None else dict(visual_plans)
    if set(plans)!={"v0","v1"}or plans["v0"]is not plan:
        raise ConstructionError("registered_v0_must_be_the_outer_solved_plan")
    if (set(plans["v1"].actors)!=set(expected.actors)or plans["v1"].sounds!=expected.sounds
            or any(not np.array_equal(np.asarray(plans["v1"].actors[k].positions_m)[:,[0,2]],expected.actors[k].positions_m[:,[0,2]])
                   or plans["v1"].actors[k].body!=expected.actors[k].body for k in expected.actors)):
        raise ConstructionError("registered_v1_must_preserve_swapped_endpoint_slots_and_events")
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True)
    directory=Path(tempfile.mkdtemp(prefix="bundle_",dir=root))
    # These are proposed camera fields, not a rewritten original measurement.
    row["listener_m"]=list(plan.camera.position_m)
    row["camera_yaw_deg"]=plan.camera.yaw_deg
    working,base,program=build_registered_scene(row,house,production_audio_record_views(records),assets,semantic,
                                              camera_evidence_policy=runtime.get("registered_camera_evidence_policy"))
    variants={"v0":deepcopy(assets),"v1":deepcopy(assets)}
    endpoint_slots=[a["endpoint_id"]for a in variants["v1"][:2]]
    variants["v1"][:2]=variants["v1"][:2][::-1]
    for asset,endpoint in zip(variants["v1"][:2],endpoint_slots):
        asset["endpoint_id"]=endpoint
    def prepare_member(key,ctx,member_request,member_plan):
        selected=variants[key]
        member_row=deepcopy(row)
        member_row["joint_prepared_visual_variant"]=key
        member_row["joint_visual_asset_endpoints"]={a["asset_id"]:a["endpoint_id"]for a in selected}
        def builder(pinned_row,unused_house,unused_records,unused_catalog,unused_semantic):
            scene=deepcopy(base)
            scene["listener_m"]=list(member_plan.camera.position_m)
            scene["camera_forward"]=member_plan.camera.forward.tolist()
            scene["camera_yaw_knots"]=[{"time_s":t,"yaw_deg":member_plan.camera.yaw_deg}for t in (0.,30.)]
            scene["joint_prepared_visual_variant"]=key
            for asset in selected:
                actor=member_plan.actors[asset["asset_id"]]
                local=np.asarray(asset["mouth_local_glb_m"],float)
                if local.shape!=(3,)or not np.isfinite(local).all():
                    raise ConstructionError("registered_actual_local_mouth_missing")
                yaw=native_yaw(actor,asset)
                mouth=np.asarray(actor.positions_m[0],float)+rotation(yaw)@local
                spec=scene["sources"][asset["endpoint_id"]]
                spec["trajectory"]=[{"time_s":t,"position_m":mouth.tolist()}for t in (0.,30.)]
                spec["body_yaw_deg"]=-math.degrees(yaw)
            member_house=deepcopy(working)
            member_house["room"]["placement"]["camera_m"]=list(member_plan.camera.position_m)
            return member_house,scene,program,selected
        return prepare_native(ctx,member_request,member_plan,
            row_factory=lambda unused:member_row,house=house,catalog=selected,records=records,
            semantic=semantic,runtime=runtime,output_dir=directory/key,
            scene_factory=builder,video_factory=NativeRegisteredVideo,static_only=True,
            family_check=(family_checks or {}).get(key),
            roles=(roles_by_visual or {}).get(key),
            minimum_visible_pixels=(minimum_visible_pixels_by_visual or {}).get(key),
            appearance_ids=appearance_ids,acoustic_preflight=True)
    result=prepare_native_bundle(context,request,plans,prepare_member,group_check=group_check,directory=directory)
    try:
        write_json(directory/"native_bundle_result.json",{
            "status":"pass"if result.payload is not None else"rejected",
            "visuals":["v0","v1"],"original_AV_members":row.get("members"),
            "program_compiled_once":True,"swap_asset_ids":ids[:2],
            "checks":[asdict(c)for c in result.checks],
            "native_full_render_and_group_QA_verified":False})
    except Exception:
        if result.close is not None:result.close()
        raise
    return result
