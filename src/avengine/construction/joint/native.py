"""Shared strict native bindings and read-only checks of the prepared scene.

No neighbour search, asset substitution or pose repair occurs in these helpers.
The existing capture owner executes the pin and retains its original support,
collision, registration, colour and mask checks.
"""
from __future__ import annotations
from copy import deepcopy
from itertools import combinations
import math
import numpy as np
from .geometry import framing_vertices
from .types import Certificate, CheckResult
from .routes import ConstructionError


def native_yaw(actor, asset):
    explicit = actor.metadata.get("native_yaw_rad")
    if explicit is not None:
        return float(explicit)
    if actor.headings_rad is None or not len(actor.headings_rad):
        raise ConstructionError("missing_planned_heading", {"entity_id": actor.entity_id})
    # Plan heading = atan2(world_dz, world_dx). Original native human/animal
    # local forward is -Z; floor devices use +X. Explicit metadata can override.
    heading = float(actor.headings_rad[0])
    return -heading-(math.pi/2 if asset.get("kind") in {"human", "performer", "animal"} else 0.)


def strict_native_bindings(plan, assets, *, meshes=None, floor_y=None, clearance_m=0., root_tolerance_m=.002):
    """Invert the original attachment anchor transform to retain exact plan roots.

    Positions are literal native body roots. Registered floor-normalized assets
    are expected; this helper never shifts a body to compensate for bad origins.
    ``meshes`` may provide already prepared local meshes by entity/asset ID.
    """
    from avengine.capture.v7_visible_mesh import indexed_static_mesh
    from avengine.capture.v7_visible_native import attachment_template
    from avengine.capture.v7_visible_attachment import rotation
    by_id = assets if isinstance(assets, dict) else {a["asset_id"]: a for a in assets}
    if set(plan.actors)-set(by_id):
        raise ConstructionError("missing_registered_asset_for_pin")
    if not 0 <= clearance_m <= .015 or not 0 <= root_tolerance_m <= .002:
        raise ValueError("strict pins retain at most the original 2mm root tolerance")
    supports, preferences, yaw_by_id = {}, {}, {}
    for key, actor in plan.actors.items():
        asset = by_id[key]
        if asset.get("attachment_surface", "floor") != "floor":
            raise ConstructionError("nonfloor_pin_requires_registered_surface_adapter", {"entity_id": key})
        mesh = meshes[key] if meshes is not None and key in meshes else indexed_static_mesh(asset["habitat_glb"])
        local = np.asarray(mesh.vertices, float)
        anchor, _, _ = attachment_template(local, "floor", asset.get("kind"), asset.get("landmarks_local_m"), mesh)
        yaw = native_yaw(actor, asset)
        root = np.asarray(actor.positions_m[0], float)
        point = root+rotation(yaw)@anchor-np.array([0., clearance_m, 0.])
        supports[key] = {"surface": "floor", "original_surface_point_m": point.tolist(),
            "normal": [0.,1.,0.], "floor_y": float(root[1] if floor_y is None else floor_y),
            "_outward_clearance_m": float(clearance_m), "yaw_rad": yaw,
            "joint_solved_pose": True, "joint_expected_root_m": root.tolist(),
            "joint_root_tolerance_m": root_tolerance_m, "joint_yaw_tolerance_rad": 1e-5,
            "joint_no_search": True}
        preferences[key], yaw_by_id[key] = root.tolist(), yaw
    return {"listener_m": list(plan.camera.position_m), "camera_yaw_deg": plan.camera.yaw_deg,
        "camera_yaw_knots": [{"time_s": t, "yaw_deg": plan.camera.yaw_deg} for t in (0., plan.duration_s)],
        "native_attachment_supports": supports, "native_placement_preferences_m": preferences,
        "joint_native_yaw_rad": yaw_by_id, "joint_strict_placement": True}


def native_world_vertices(entry):
    """Current original posed mesh transformed by the actual native root readback."""
    from avengine.capture.v7_visible_mesh import indexed_static_mesh
    if entry.get("motion_episode"):
        path = entry.get("motion_pose_path")
        if not path:
            raise ValueError("moving native actor has no current pose mesh")
        mesh = indexed_static_mesh(path)
    else:
        mesh = entry["original_mesh"]
    transform = np.asarray(entry["object"].root_scene_node.absolute_transformation(), float)
    local = np.asarray(mesh.vertices, float)
    return local@transform[:3,:3].T+transform[:3,3], transform


def native_rest_framing(video, entry):
    vertices, _ = native_world_vertices(entry)
    sensor = video.sim._sensors["rgb"]._sensor_object
    projection = np.asarray(sensor.render_camera.projection_matrix, float)
    view = np.linalg.inv(np.asarray(sensor.node.absolute_transformation(), float))
    state, margin = framing_vertices(vertices, view, projection, video.width, video.height)
    return {"state": state, "edge_margin_px": margin if math.isfinite(margin) else None,
            "original_mesh_vertices": len(vertices), "authority": "actual_native_full_mesh_projection",
            "occlusion_is_not_out_of_view": True}


def native_static_checks(video, scene, plan, *, roles=None, minimum_visible_pixels=None,
                         appearance_ids=(), standing_checks=None, root_tolerance_m=.002):
    """Check an already prepared static scene without moving any entity.

    Actual masks and colour use frame zero of the final complete scene. This
    one-frame existence witness does not certify a dynamic universal condition.
    """
    if any(not np.array_equal(a.positions_m, np.repeat(np.asarray(a.positions_m)[:1], len(a.positions_m), axis=0))
           for a in plan.actors.values()):
        raise ConstructionError("static_oracle_received_motion")
    entries = {e["declaration"]["asset_id"]:e for e in video.actors}
    checks=[]
    def add(name, ok, reason="", **evidence):
        checks.append(CheckResult(name, "pass" if ok else "fail", reason, evidence=evidence))
    add("native:identities", set(entries)==set(plan.actors), "exact_registered_entity_set")
    if set(entries)!=set(plan.actors):
        return Certificate(tuple(checks), "native_static_preflight")
    roles = roles or plan.metadata.get("roles", {})
    minimum_visible_pixels = dict(minimum_visible_pixels or {})
    counts = video._native_pair_pixels(scene, list(entries.values()))
    sensor = video.sim._sensors["rgb"]._sensor_object
    camera_matrix = np.asarray(sensor.node.absolute_transformation(), float)
    camera_delta = float(np.linalg.norm(camera_matrix[:3,3]-plan.camera.position_m))
    forward_error = float(np.linalg.norm(-camera_matrix[:3,2]-plan.camera.forward))
    projection = np.asarray(sensor.render_camera.projection_matrix, float)
    hfov = math.degrees(2*math.atan(1/float(projection[0,0])))
    add("native:camera_intrinsics", video.width==plan.camera.width and video.height==plan.camera.height
        and abs(hfov-plan.camera.hfov_deg)<1e-4, actual_width=video.width, actual_height=video.height,
        actual_hfov_deg=hfov)
    add("native:camera_unchanged", camera_delta <= 1e-4 and forward_error <= 1e-5,
        position_error_m=camera_delta, forward_vector_error=forward_error)
    actual_roots={}
    for key, entry in entries.items():
        actor=plan.actors[key];_,transform=native_world_vertices(entry)
        actual_roots[key]=transform[:3,3]
        error=float(np.linalg.norm(transform[:3,3]-np.asarray(actor.positions_m[0])))
        wanted_yaw=native_yaw(actor,entry["declaration"])
        observed_yaw=math.atan2(transform[0,2],transform[0,0])
        yaw_error=abs((observed_yaw-wanted_yaw+math.pi)%(2*math.pi)-math.pi)
        add("native:"+key+":exact_pose", error <= root_tolerance_m+1e-7 and yaw_error <= 1e-5,
            "execution_must_not_replan", root_error_m=error, yaw_error_rad=yaw_error,
            actual_root_m=transform[:3,3].tolist(), expected_root_m=np.asarray(actor.positions_m[0]).tolist())
        framing=native_rest_framing(video,entry);wanted=roles.get(key,"any")
        add("native:"+key+":rest_framing", framing["state"] in ({"inside","outside"}if wanted=="any"else{wanted}),
            **framing)
        if key in minimum_visible_pixels:
            pixels=int(counts.get(key,0));limit=int(minimum_visible_pixels[key])
            add("native:"+key+":visible_pixels", pixels>=limit, actual_visible_pixels=pixels, required_pixels=limit)
        if actor.body.species=="human":
            distance=float(np.linalg.norm((actual_roots[key]-camera_matrix[:3,3])[[0,2]]))
            add("native:"+key+":camera_distance", distance>=1.5-1e-6, actual_m=distance)
    for a,b in combinations(actual_roots,2):
        if plan.actors[a].body.species==plan.actors[b].body.species=="human":
            distance=float(np.linalg.norm((actual_roots[a]-actual_roots[b])[[0,2]]))
            add("native:"+a+":"+b+":spacing", distance>=1.-1e-6, actual_m=distance)
    support = standing_checks if standing_checks is not None else getattr(video,"attachment_checks",[])
    passed={r.get("asset_id")for r in support if r.get("status")=="pass"}
    add("native:original_support_checks", set(entries)<=passed, passed_asset_ids=sorted(str(v)for v in passed))
    if appearance_ids:
        obs=video.sim.get_sensor_observations()
        lengths={name:len(getattr(video,name,[]))for name in ("color_checks","material_checks","phase2_chair_pixel_checks")}
        try:
            video._check_source_frame(obs["rgb"][:,:,:3],obs["semantic"],0)
            observed=video.color_checks[lengths["color_checks"]:]
            matched={r["asset_id"]for r in observed if r.get("matched")}
            for key in appearance_ids:
                add("native:"+key+":appearance", key in matched,
                    measurements=[r for r in observed if r.get("asset_id")==key])
        finally:
            for name,length in lengths.items():
                if hasattr(video,name):
                    del getattr(video,name)[length:]
    return Certificate(tuple(checks), "native_static_preflight")


class PreparedStaticScene:
    """The exact live prepared native world. The successful caller owns close()."""
    def __init__(self, video, scene, row, house, assets, program, directory, certificate):
        self.video, self.scene, self.row = video, scene, row
        self.house, self.assets, self.program = house, assets, program
        self.directory, self.certificate = directory, certificate
        self.closed = False

    def bind_solution(self, record):
        from avengine.timeline.continuous_pilot import write_json
        self.row["joint_solution"] = deepcopy(record)
        self.scene["joint_solution"] = deepcopy(record)
        self.scene.setdefault("production_scenario",{})["joint_solution"] = deepcopy(record)
        write_json(self.directory/"joint_solution.json",record)

    def close(self):
        if not self.closed:
            self.video.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def prepare_native(context, request, plan, *, row_factory, house, catalog, records,
                          semantic, runtime, output_dir, family_check=None, roles=None,
                          minimum_visible_pixels=None, appearance_ids=(), static_only=False):
    """A solve(..., prepare_and_verify=...) oracle: prepare one exact pin, never search.

    row_factory(plan) creates only family-specific row/asset/sound fields. This
    shared adapter applies the inverse-anchor pins, runs the original native
    physical checks once, then shared readbacks and optional family-only checks.
    A passing live payload is retained for rendering; a failed one is closed.
    """
    import tempfile
    from pathlib import Path
    from dataclasses import asdict
    from .types import PreparedCandidate
    from avengine.timeline.v7_family_clock import build_constructed_scene
    from avengine.capture.v7_phase2_native import NativePhase2Video
    from avengine.timeline.continuous_pilot import write_json
    if static_only and any(np.any(np.diff(np.asarray(a.positions_m),axis=0)) for a in plan.actors.values()):
        raise ConstructionError("static_oracle_received_motion")
    root = Path(output_dir)
    root.mkdir(parents=True,exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="candidate_",dir=root))
    video = None
    try:
        row = deepcopy(row_factory(plan))
        if "row" in row and "asset_ids" not in row:
            row = row["row"]
        selected = {a["asset_id"]:a for a in catalog if a["asset_id"] in plan.actors}
        pins = strict_native_bindings(plan,selected,floor_y=context.room["measured_floor_y_m"])
        row.update(pins)
        row["joint_planned_sound_events"] = [asdict(e)for e in plan.sounds]
        working,scene,program,assets = build_constructed_scene(row,house,records,catalog,semantic)
        # Direct scene fields too: legacy row adapters may not forward new flags.
        scene.update(deepcopy(pins))
        native_runtime = deepcopy(runtime)
        # Keep caller taskset authority; never restore another task's fixed RIR CPUs.
        import os
        native_runtime.pop("fresh_rir_cpu_ids",None)
        cores = len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else (os.cpu_count() or 1)
        native_runtime["fresh_rir_workers"] = min(int(native_runtime.get("fresh_rir_workers",1)),max(1,cores-1))
        native_runtime["visible_placement_max_proposals_per_asset"] = 1
        native_runtime["visible_placement_max_support_candidates_per_asset"] = 1
        native_runtime["source_material_cache_directory"] = str(root/"source_materials")
        video = NativePhase2Video(working,scene,native_runtime,assets,
                    source_material_cache_directory=native_runtime["source_material_cache_directory"])
        standing = video.qualify_standing(scene,directory/"standing")
        common = native_plan_checks(video,scene,plan,roles=roles,
                    minimum_visible_pixels=minimum_visible_pixels,appearance_ids=appearance_ids,
                    standing_checks=standing)
        extra = tuple(family_check(video,scene,video.actors)) if family_check is not None and common.passed else ()
        if common.passed and request.options.get("require_native_preflight") and not extra:
            extra = (CheckResult("native:family_obligations","unknown","family_native_checks_not_supplied"),)
        certificate = Certificate(common.checks+extra,"native_static_preflight")
        write_json(directory/"native_certificate.json",asdict(certificate))
        write_json(directory/"strict_row.json",row)
        if not certificate.passed:
            video.close()
            return PreparedCandidate(None,certificate.checks)
        payload = PreparedStaticScene(video,scene,row,working,assets,program,directory,certificate)
        return PreparedCandidate(payload,certificate.checks,payload.close)
    except (ConstructionError,ValueError) as exc:
        if video is not None:
            video.close()
        check = CheckResult("native:execution_conflict","fail",str(exc),
                            evidence={"directory":str(directory),"exception_type":type(exc).__name__,
                                      "native_search_allowed":False})
        write_json(directory/"native_conflict.json",asdict(check))
        return PreparedCandidate(None,(check,))
    except Exception:
        if video is not None:
            video.close()
        raise


def native_plan_checks(video, scene, plan, **kwargs):
    """Match every prepared native root/yaw sample to the selected plan clock."""
    from dataclasses import replace
    frozen={key:replace(actor,positions_m=np.repeat(np.asarray(actor.positions_m)[:1],len(actor.positions_m),axis=0))
            for key,actor in plan.actors.items()}
    common=native_static_checks(video,scene,replace(plan,actors=frozen),**kwargs)
    checks=list(common.checks)
    entries={e["declaration"]["asset_id"]:e for e in video.actors}
    for key,actor in plan.actors.items():
        if key not in entries:
            continue
        p=np.asarray(actor.positions_m,float);entry=entries[key]
        moving=bool(np.any(np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-7))
        turning=actor.headings_rad is not None and bool(np.ptp(np.unwrap(actor.headings_rad))>1e-5)
        if not moving and not turning:
            continue
        record=entry.get("motion_episode",{})
        readbacks=record.get("native_30Hz_readbacks",[])
        matrices=np.asarray([r.get("body_to_world_m",[])for r in readbacks],float)
        actual=matrices[:,:3,3] if matrices.shape==(len(p),4,4) else np.empty((0,3))
        indices_ok=(not readbacks or not all("action_sample_index"in r for r in readbacks)
                    or [int(r["action_sample_index"])for r in readbacks]==list(range(len(p))))
        clock_ok=(actual.shape==p.shape and np.isfinite(actual).all() and indices_ok
                  and float(record.get("sample_rate_hz",0))==plan.frame_rate_hz)
        if not clock_ok:
            checks.append(CheckResult("native:"+key+":motion_clock","fail","native root clock differs or is missing"))
            continue
        error=float(np.max(np.linalg.norm(actual-p,axis=1)))
        checks.append(CheckResult("native:"+key+":all_root_samples","pass"if error<=.002+1e-7 else"fail",
            "native_execution_must_not_replan",evidence={"frames":len(p),"maximum_root_error_m":error,"authority":"native_30Hz_body_to_world_readbacks"}))
        if actor.headings_rad is not None:
            first_native=native_yaw(actor,entry["declaration"])
            if actor.metadata.get("native_yaw_rad_per_frame") is not None:
                expected=np.asarray(actor.metadata["native_yaw_rad_per_frame"],float)
            else:
                expected=first_native-(np.asarray(actor.headings_rad)-actor.headings_rad[0])
            actual_yaw=np.arctan2(matrices[:,0,2],matrices[:,0,0])
            ok=(actual_yaw.shape==expected.shape==(len(p),) and np.isfinite(actual_yaw).all() and np.isfinite(expected).all())
            error=float(np.max(np.abs((actual_yaw-expected+math.pi)%(2*math.pi)-math.pi)))if ok else math.inf
            checks.append(CheckResult("native:"+key+":all_yaw_samples","pass"if error<=1e-5 else"fail",
                "native_execution_must_not_change_planned_heading",evidence={"frames":len(p),"maximum_yaw_error_rad":error if math.isfinite(error)else None}))
    return Certificate(tuple(checks),"native_prepared_plan_readback")


def prepare_static_native(context, request, plan, **kwargs):
    return prepare_native(context,request,plan,static_only=True,**kwargs)


PreparedNativeScene = PreparedStaticScene
