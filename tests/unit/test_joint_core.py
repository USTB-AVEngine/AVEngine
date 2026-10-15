"""Counterexamples for the shared planner; no QA truth or scorer modifications."""
import math
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from avengine.rooms.walkable_map import WalkableMap, VERSION, SPECIES_RADII_M
from avengine.construction.joint import *
from avengine.construction.joint.routes import minimum_track_distance

pytestmark = pytest.mark.fast_unit


class FlatNav:
    def snap_point(self, p):
        return np.asarray(p)


def context(tmp_path=None, holes=()):
    mask = np.ones((180, 180), bool)
    for z, x in holes:
        mask[z, x] = False
    meta = {"version": VERSION, "species_radii_m": SPECIES_RADII_M,
            "footprint_policy": "independent-radius-coverage-v2", "room_id": "fixture"}
    m = WalkableMap([-4.5, -7.5], .05, 0., {k: mask.copy() for k in SPECIES_RADII_M}, meta)
    room = {"room_label": "fixture", "floor_polygon": {"type": "Polygon", "coordinates": [
        [[-4.5, -7.5], [4.5, -7.5], [4.5, 1.5], [-4.5, 1.5], [-4.5, -7.5]]]},
        "measured_floor_y_m": 0., "placement": {"camera_m": [0., 1.25, .5]}}
    c = RoomContext(room, walkable_map=m, pathfinder=FlatNav(), cache_dir=tmp_path,
                    camera_limit=6, grid_step_m=.5)
    c.cameras = (CameraPose((0., 1.25, .5), 0.),)
    return c


def depth(camera, z=np.inf, source="visual_mesh"):
    return DepthBuffer(camera, np.full((camera.height, camera.width), z), source=source)


@pytest.mark.parametrize("yaw", [0., 90., -135., 179.])
def test_projection_sign_rotation_and_focal(yaw):
    c = CameraPose((1., 1.25, 2.), yaw)
    p = np.array(c.position_m)+4*c.forward+c.right
    uv, z = project_points(c, [p])
    assert uv[0, 0] > c.width/2 and z[0] == pytest.approx(4)
    assert uv[0, 1] == pytest.approx(c.height/2)


def test_horizontal_center_does_not_hide_cropped_feet():
    c = CameraPose((0., 1.5, 0.), 0.)
    assert framing(c, [0, 0, -1.6])[0] == "edge"
    assert framing(c, [0, 0, -4.])[0] == "inside"
    assert framing(c, [8, 0, -4.])[0] == "outside"


def test_missing_depth_is_unknown_except_wholly_outside():
    c = CameraPose((0., 1.25, 0.), 0.)
    assert predict_visibility(c, [0, 0, -4]).state == "unknown"
    assert predict_visibility(c, [8, 0, -4]).state == "out_of_view"


def test_zero_pixels_occlusion_is_not_out_of_view_and_proxy_not_real_lower_bound():
    c = CameraPose((0., 1.25, 0.), 0.)
    hidden = predict_visibility(c, [0, 0, -4], depth=depth(c, 2.))
    assert hidden.state == "fully_occluded"
    assert hidden.confidence == "outer_box_depth_margin"
    seen = predict_visibility(c, [0, 0, -4], depth=depth(c))
    assert seen.state == "visible_clear" and seen.visible_pixels_estimate > 100
    assert seen.visible_pixels_lower == 0
    assert seen.evidence["inner_proxy_visible_pixels"] > 100


def test_depth_holes_and_acoustic_proxies_cannot_prove_occlusion():
    c = CameraPose((0., 1.25, 0.), 0.)
    d = depth(c, 2.)
    d.depth_m[170:190, 310:330] = np.nan
    assert predict_visibility(c, [0, 0, -4], depth=d).state == "unknown"
    assert predict_visibility(c, [0, 0, -4], depth=depth(c, 2., "acoustic_proxy_mesh")).state == "unknown"


def test_registered_actor_proxy_reports_identity_but_not_full_occlusion_proof():
    c = CameraPose((0., 1.25, 0.), 0.)
    p = predict_visibility(c, [0, 0, -4.5], depth=depth(c),
         occluders=(("speaker", [0, 0, -3], BodySpec("slide", .25, 1.)),))
    assert p.state == "visible_occluded"
    assert p.occluder_ids == ("speaker",)
    assert .1 < p.occluded_fraction < .9
    assert p.evidence["proxy_occluders"]


def test_real_triangle_hole_stays_visible_and_forward_depth_not_range():
    import trimesh
    c = CameraPose((0., 1.25, 0.), 0., width=160, height=90)
    mesh = trimesh.creation.box((8., 4., .1)); mesh.apply_translation([0, 1., -2.])
    ray = MeshRaycaster(mesh, face_ids=["wall"]*len(mesh.faces))
    d = build_depth_buffer(c, ray)
    assert np.allclose(d.depth_m[20:60, 20:140], 1.95, atol=1e-5)
    p = predict_visibility(c, [0, 0, -4], depth=d)
    assert p.state == "fully_occluded" and p.occluder_ids == ("wall",)


def test_continuous_pair_distance_catches_between_frame_collision():
    a = np.array([[-1., 0, 0], [1., 0, 0]])
    b = -a
    assert minimum_track_distance(a, b) == pytest.approx(0)


def test_map_supercover_catches_thin_blocked_cell():
    c = context(holes=[(90, 90)])
    a, b = [-.2, 0, -3.], [.2, 0, -3.]
    with pytest.raises(ConstructionError, match="segment"):
        validate_track(c, [a, b])


def test_astar_goes_around_obstacle_and_budget_is_explicit():
    c = context(holes=[(z, 90) for z in range(40, 130)])
    a, b = [-1, 0, -3], [1, 0, -3]
    path = route_between(c, a, b)
    assert len(path) > 2
    assert all(c.walkable_map.segment_free(x[[0,2]], y[[0,2]]) for x, y in zip(path[:-1], path[1:]))
    with pytest.raises(ConstructionError, match="search_budget"):
        route_between(c, a, b, max_expansions=1)


@pytest.mark.parametrize("species,speed", [("human", 1.), ("dog", .7), ("cat", .5), ("slide", .3)])
def test_rounded_route_arclength_speed_and_full_track(species, speed):
    c = context()
    p = construct_route(c, [[-2, 0, -4], [0, 0, -4], [0, 0, -6]],
                        species=species, speed_m_s=speed, camera=c.cameras[0])
    assert p.positions_m.shape == (901, 3)
    assert 3 < p.metadata["route_length_m"] < 4
    v = np.linalg.norm(np.diff(p.positions_m[:, [0,2]], axis=0), axis=1)*30
    assert v.max() == pytest.approx(speed, abs=.003)
    np.testing.assert_allclose(p.positions_m[-1], [0,0,-6])
    assert p.metadata["validation"]["map_full_frames"] == 901


def test_dwell_forces_stop_and_second_leg_after_dwell():
    c = context()
    p = construct_route(c, [[-2,0,-4], [0,0,-4], [0,0,-6]], dwell_s=[0,1.,0])
    intervals = p.metadata["walk_intervals_s"]
    assert intervals[1][0]-intervals[0][1] == pytest.approx(1.)
    at = round((intervals[0][1]+.5)*30)
    np.testing.assert_allclose(p.positions_m[at], [0,0,-4])


def test_species_speed_floor_snap_and_parent_duration_rejected():
    c = context()
    with pytest.raises(ConstructionError, match="speed_band"):
        construct_route(c, [[0,0,-3], [0,0,-4]], species="cat", speed_m_s=1.)
    with pytest.raises(ConstructionError, match="duration"):
        construct_route(c, [[0,0,-3], [0,0,-6]], duration_s=1.)
    with pytest.raises(ConstructionError, match="changes_floor"):
        validate_track(c, [[0,3,-3]])


def test_certification_rejects_missing_nav_and_multiple_human_sound_classes():
    c = context()
    a = static_actor("a", [0,0,-4])
    p = ScenePlan(c.cameras[0], {"a": a})
    assert certify_plan(c,p).passed
    p.sounds = (SoundEvent("a","s1","speech",1,2),SoundEvent("a","s2","cough",4,5))
    assert not certify_plan(c,p).passed
    c.pathfinder = None
    assert any(x.status == "unknown" for x in certify_plan(c,p).checks)


def test_grid_cache_reused_and_changed_camera_does_not_reuse(tmp_path):
    c = context(tmp_path)
    g = c.grid(c.cameras[0])
    c2 = context(tmp_path)
    g2 = c2.grid(c2.cameras[0])
    np.testing.assert_array_equal(g.frustum_state,g2.frustum_state)
    assert c2.cache_hits == 1 and c2.cache_misses == 0
    c2.grid(replace(c2.cameras[0],yaw_deg=15.))
    assert c2.cache_misses == 1


def test_solver_no_checks_not_certified_and_plugin_bug_not_infeasibility():
    c = context()
    p = ScenePlan(c.cameras[0], {"a": static_actor("a", [0,0,-4])})
    plugin = SimpleNamespace(candidates=lambda c,r: iter([p]), verify=lambda c,r,p: ())
    result = solve(c, FamilyRequest("QA-21","class",("a",)), plugin)
    assert result.status == "unavailable" and result.plan is None
    plugin.verify = lambda c,r,p: [CheckResult("branch", "pass")]
    assert solve(c, FamilyRequest("QA-21","class",("a",)), plugin).status == "certified"
    plugin.verify = lambda c,r,p: (_ for _ in ()).throw(RuntimeError("plugin bug"))
    with pytest.raises(RuntimeError,match="plugin bug"):
        solve(c, FamilyRequest("QA-21","class",("a",)), plugin)


def test_static_role_layout_constructs_inside_outside_without_human_collision():
    c = context()
    bodies = {"target":BodySpec(), "other":BodySpec()}
    plans = list(enumerate_static_layouts(c, bodies, {"target":"inside","other":"outside"}, max_candidates=2))
    assert len(plans) == 2
    for p in plans:
        assert certify_plan(c,p).passed
        assert framing(p.camera,p.actors["target"].positions_m[0])[0] == "inside"
        assert framing(p.camera,p.actors["other"].positions_m[0])[0] == "outside"


def test_exact_mesh_refinement_keeps_occluder_hole_and_identity():
    import trimesh
    c = CameraPose((0.,1.25,0.),0.,width=160,height=90)
    target_mesh=trimesh.creation.box((.6,1.8,.2))
    t=np.eye(4);t[:3,3]=[0,.9,-4]
    target=MeshInstance("target",MeshRaycaster(target_mesh),t)
    pieces=[]
    for x in (-.5,.5):
        m=trimesh.creation.box((.6,3.,.1));m.apply_translation([x,1.,-2]);pieces.append(m)
    barrier=MeshInstance("with_gap",MeshRaycaster(trimesh.util.concatenate(pieces)))
    p=predict_mesh_visibility(c,target,scene_depth=depth(c),occluders=(barrier,))
    assert p.visible_pixels_estimate > 0
    assert p.state != "fully_occluded"
    full=trimesh.creation.box((3.,3.,.1));full.apply_translation([0,1.,-2])
    b=MeshInstance("solid",MeshRaycaster(full))
    p=predict_mesh_visibility(c,target,scene_depth=depth(c),occluders=(b,))
    assert p.state == "fully_occluded" and p.occluder_ids == ("solid",)
    assert p.evidence["target_only_pixels"] > 0
    assert not p.evidence["native_pixel_truth"]


def test_exact_mesh_scaled_transform_depth_and_target_centroid():
    import trimesh
    c=CameraPose((0.,1.25,0.),0.,width=160,height=90)
    mesh=trimesh.creation.box((.6,1.8,.2))
    t=np.eye(4);t[:3,:3]*=.5;t[:3,3]=[-1.,.45,-4.]
    target=MeshInstance("t",MeshRaycaster(mesh),t)
    d=raster_mesh_depth(c,target)
    assert np.min(d) == pytest.approx(3.95,abs=1e-5)
    p=predict_mesh_visibility(c,target,scene_depth=depth(c))
    assert p.state == "visible_clear"
    assert p.evidence["target_centroid_xy_px"][0] < c.width/2
    with pytest.raises(ValueError,match="own occluders"):
        predict_mesh_visibility(c,target,scene_depth=depth(c),occluders=(target,))


def test_depth_cache_survives_new_context_and_new_registered_occluder_isolated(tmp_path):
    import trimesh
    c=context(tmp_path)
    wall=trimesh.creation.box((9.,4.,.1));wall.apply_translation([0,1.,-6.])
    c.raycaster=MeshRaycaster(wall,evidence={"fixture":"wall"})
    c.identity["geometry"]={"fixture":"wall"};c.identity["geometry_source"]="visual_mesh"
    d=c.depth(c.cameras[0]);assert c.cache_misses==1
    c._depth={};c.raycaster.first_hit=lambda *a: (_ for _ in ()).throw(AssertionError("cache missed"))
    np.testing.assert_allclose(c.depth(c.cameras[0]).depth_m,d.depth_m)
    assert c.cache_hits==1
    obj=trimesh.creation.box((1.,2.,.1));obj.apply_translation([0,1.,-2.])
    oc=MeshInstance("registered",MeshRaycaster(obj,evidence={"fixture":"foreground"}))
    c.raycaster=MeshRaycaster(wall,evidence={"fixture":"wall"})
    altered=c.with_occluders(oc)
    assert altered._owned_video is None
    d2=altered.depth(c.cameras[0])
    assert d2.depth_m[c.cameras[0].height//2,c.cameras[0].width//2] < d.depth_m[c.cameras[0].height//2,c.cameras[0].width//2]
    assert altered._key(c.cameras[0]) != c._key(c.cameras[0])


def test_scheduler_measured_activity_respects_silent_query_and_moving_window():
    req=SoundRequirement("a","clip","speech",((.2,1.2),),(0.,5.),((2.,4.),))
    r=schedule_sound_events([req],silent_intervals_s=((4.,5.),))
    assert r.status=="solved"
    e=r.events[0]
    assert 2. <= e.start_s < e.end_s <= 3.5
    assert e.wet_end_s <= 4.
    assert e.end_s-e.start_s == pytest.approx(1.)


def test_scheduler_backtracks_starts_and_does_not_overlap_same_voice():
    a=SoundRequirement("a","one","speech",((0.,1.),),(0.,4.))
    b=SoundRequirement("a","two","speech",((0.,1.),),(0.,0.))
    r=schedule_sound_events([a,b])
    assert r.status=="solved" and r.starts_s[0] >= 1.5 and r.starts_s[1] == 0.
    assert len(r.events)==2


def test_scheduler_zero_domain_and_budget_are_different():
    a=SoundRequirement("a","one","speech",((0.,1.),),(0.,0.))
    r=schedule_sound_events([a],silent_intervals_s=((0.,3.),))
    assert r.status=="unavailable" and r.reasons[0].startswith("empty_start_domain")
    b=SoundRequirement("a","two","speech",((0.,1.),),(0.,4.))
    r=schedule_sound_events([a,b],max_nodes=1)
    assert r.status=="budget_exhausted"


def test_navmesh_drift_uses_original_native_two_millimetre_limit():
    c=context()
    c.pathfinder=SimpleNamespace(snap_point=lambda p:np.asarray(p)+[.01,0,0])
    with pytest.raises(ConstructionError,match="snap_tolerance"):
        validate_track(c,[[0,0,-3]])


def test_camera_search_includes_interior_when_inset_perimeter_blocked():
    c=context()
    for layer in c.walkable_map.layers.values():
        layer[:]=False
        layer[65:115,65:115]=True
    cameras=candidate_cameras(c.room,c.walkable_map,c.pathfinder,limit=6)
    assert cameras
    for camera in cameras:
        assert c.walkable_map.contains(np.asarray(camera.position_m)[[0,2]],"human")


def test_elevated_navmesh_does_not_float_actor_or_remove_legal_camera():
    c=context()
    c.pathfinder=SimpleNamespace(snap_point=lambda p:np.asarray(p)+[0,.166,0])
    q,report=validate_track(c,[[0,0,-3],[0,0,-4]])
    np.testing.assert_array_equal(q[:,1],[0,0])
    assert report["navmesh_y_offset_range_m"] == pytest.approx([.166,.166])
    assert candidate_cameras(c.room,c.walkable_map,c.pathfinder,limit=6)


def native_projection(width=640,height=360):
    f=1/math.tan(math.radians(42.5));near=.01;far=100.
    return np.array([[f,0,0,0],[0,f*width/height,0,0],
                     [0,0,-(far+near)/(far-near),-2*far*near/(far-near)],[0,0,-1,0]])


def test_native_vertices_framing_catches_vertical_crop_and_multiplane_crossing():
    from avengine.construction.joint.geometry import body_box
    camera=np.eye(4);camera[:3,3]=[0,1.5,0]
    view=np.linalg.inv(camera);p=native_projection()
    assert framing_vertices(body_box([0,0,-4],BodySpec()),view,p,640,360)[0]=="inside"
    assert framing_vertices(body_box([0,0,-1.6],BodySpec()),view,p,640,360)[0]=="edge"
    assert framing_vertices(body_box([8,0,-4],BodySpec()),view,p,640,360)[0]=="outside"
    assert framing_vertices([[-10,0,-2],[10,0,-2],[0,10,-2]],view,p,640,360)[0]=="edge"


def test_strict_native_pin_inverts_offcentre_attachment_anchor():
    import trimesh
    from avengine.capture.v7_visible_native import attachment_template
    from avengine.capture.v7_visible_attachment import support_points
    mesh=trimesh.creation.box((.6,1.8,.4));mesh.apply_translation([.2,.9,.13])
    actor=static_actor("a",[1.,0.,-4.],facing_rad=.3)
    plan=ScenePlan(CameraPose((0.,1.25,0.),0.),{"a":actor})
    asset={"asset_id":"a","kind":"human","attachment_surface":"floor"}
    bindings=strict_native_bindings(plan,{"a":asset},meshes={"a":mesh},floor_y=0.,clearance_m=.002)
    pin=bindings["native_attachment_supports"]["a"]
    assert pin["joint_solved_pose"] and pin["joint_no_search"]
    meta={**asset,"_attachment_template":attachment_template(np.asarray(mesh.vertices),"floor","human",mesh=mesh)}
    support={**pin,"point":(np.array(pin["original_surface_point_m"])+np.array(pin["normal"])*pin["_outward_clearance_m"]).tolist()}
    root,yaw,_,_=support_points(meta,support)
    np.testing.assert_allclose(root,actor.positions_m[0],atol=1e-12)
    assert yaw==pytest.approx(-.3-math.pi/2)


def fake_native_world(plan, offset=(0,0,0), yaw=0.):
    import trimesh
    mesh=trimesh.creation.box((.6,1.8,.4));mesh.apply_translation([0,.9,0])
    actor=plan.actors["a"];actor.metadata["native_yaw_rad"]=0.
    from avengine.capture.v7_visible_attachment import rotation
    transform=np.eye(4);transform[:3,:3]=rotation(yaw);transform[:3,3]=actor.positions_m[0]+offset
    root_node=SimpleNamespace(absolute_transformation=lambda:transform)
    entry={"object":SimpleNamespace(root_scene_node=root_node),"original_mesh":mesh,
           "declaration":{"asset_id":"a","kind":"human"}}
    camera=np.eye(4);camera[:3,3]=plan.camera.position_m
    sensor=SimpleNamespace(render_camera=SimpleNamespace(projection_matrix=native_projection()),
                           node=SimpleNamespace(absolute_transformation=lambda:camera))
    sim=SimpleNamespace(_sensors={"rgb":SimpleNamespace(_sensor_object=sensor)})
    video=SimpleNamespace(actors=[entry],sim=sim,width=640,height=360,
          _native_pair_pixels=lambda scene,entries:{"a":2000},attachment_checks=[{"asset_id":"a","status":"pass"}])
    return video,entry


def test_native_common_readbacks_reject_repositioning_or_changed_yaw():
    p=ScenePlan(CameraPose((0.,1.25,0.),0.),{"a":static_actor("a",[0,0,-4])})
    v,e=fake_native_world(p)
    assert native_static_checks(v,{},p,roles={"a":"inside"},minimum_visible_pixels={"a":100}).passed
    v,e=fake_native_world(p,offset=(.03,0,0))
    assert not native_static_checks(v,{},p).passed
    v,e=fake_native_world(p,yaw=.1)
    assert not native_static_checks(v,{},p).passed


def test_solver_native_failure_backtracks_and_success_retains_same_prepared_payload():
    c=context();p=ScenePlan(c.cameras[0],{"a":static_actor("a",[0,0,-4])})
    plugin=SimpleNamespace(candidates=lambda c,r:iter([p,p]),verify=lambda c,r,p:[CheckResult("branch","pass")])
    calls=[];closed=[];payload=object()
    def prepare(c,r,p):
        calls.append(p)
        i=len(calls)
        return PreparedCandidate(payload,(CheckResult("native:mask","fail"if i==1 else"pass"),),lambda:closed.append(i))
    result=solve(c,FamilyRequest("QA-21","sound",("a",),{"require_native_preflight":True}),plugin,prepare_and_verify=prepare)
    assert result.status=="certified" and len(calls)==2 and closed==[1]
    assert result.prepared.payload is payload
    with result.prepared as live:
        assert live is payload
    assert closed==[1,2]
    blocked=solve(c,FamilyRequest("QA-21","sound",("a",),{"require_native_preflight":True}),plugin)
    assert blocked.status=="unavailable" and blocked.reasons==("required_native_preflight_not_supplied",)


def test_fixed_clock_straight_arc_preserves_pause_and_compact_depth():
    c=context();t=np.arange(901)/30
    d=1.02*np.clip(t-1,0,2)+1.02*np.clip(t-6,0,2)
    p=straight_then_arc(c,[0,0,-4.8],[.5,-math.sqrt(.75)],d,2.04,
                       turn_radius_m=.8,turn_sign=1,camera=c.cameras[0])
    np.testing.assert_allclose(p.positions_m[135],[0,0,-4.8],atol=1e-8)
    assert p.positions_m[-1,0] == pytest.approx(1.49,abs=.02)
    assert p.positions_m[:,2].min() > -5.3
    np.testing.assert_array_equal(p.metadata["distance_clock_m"],d)
    assert p.metadata["validation"]["map_full_frames"]==901


def test_repeat_bouts_reverses_same_curve_and_holds_after_two():
    c=context()
    original=construct_route(c,[[-2,0,-4],[0,0,-4],[0,0,-6]],entity_id="a")
    repeated=repeat_bouts(original,(0,10),context=c,camera=c.cameras[0])
    np.testing.assert_allclose(repeated.positions_m[-1],original.positions_m[0])
    np.testing.assert_allclose(repeated.positions_m[600:],np.repeat(original.positions_m[:1],301,axis=0))
    distance=np.linalg.norm(repeated.positions_m[300:601,None,:]-original.positions_m[None,:301,:],axis=2).min(axis=1)
    assert distance.max()<.04
    assert len(repeated.metadata["stationary_turn_intervals_s"])==1
    a,b=repeated.metadata["stationary_turn_intervals_s"][0]
    turn=repeated.positions_m[math.ceil(a*30):math.floor(b*30)+1]
    assert np.max(np.linalg.norm(turn-turn[0],axis=1))<1e-9
    assert certify_plan(c,ScenePlan(c.cameras[0],{"a":repeated})).passed


def test_static_product_budget_varies_all_roles_and_prunes_bad_nav_points():
    c=context()
    bodies={"a":BodySpec(),"b":BodySpec()}
    plans=list(enumerate_static_layouts(c,bodies,{"a":"inside","b":"outside"},max_candidates=8))
    assert len({tuple(p.actors["a"].positions_m[0])for p in plans})>1
    assert len({tuple(p.actors["b"].positions_m[0])for p in plans})>1
    c.pathfinder=SimpleNamespace(snap_point=lambda p:np.asarray(p)+([.1,0,0]if p[0]>0 else[0,.17,0]))
    plans=list(enumerate_static_layouts(c,bodies,{"a":"inside","b":"outside"},max_candidates=8))
    assert plans and all(all(a.positions_m[0,0]<=0 for a in p.actors.values())for p in plans)


def test_native_motion_readback_requires_same_complete_clock_and_roots():
    c=context();actor=construct_route(c,[[0,0,-3],[0,0,-5]],entity_id="a")
    actor.metadata["native_yaw_rad"]=0.
    p=ScenePlan(CameraPose((0.,1.25,0.),0.),{"a":actor})
    v,e=fake_native_world(p)
    matrices=np.repeat(np.eye(4)[None],901,axis=0);matrices[:,:3,3]=actor.positions_m
    e["motion_episode"]={"world_roots_m":actor.positions_m.tolist(),"sample_rate_hz":30.,
                          "status":"pass","final_complete_scene_recheck":{"status":"pass","method":"unit fixture"},
                          "native_30Hz_readbacks":[{"body_to_world_m":m.tolist(),"action_sample_index":i}for i,m in enumerate(matrices)]}
    # Actual posed mesh loading belongs to the native adapter; a retained static
    # local mesh suffices for this readback-clock counterexample.
    import avengine.construction.joint.native as native
    original=native.native_world_vertices
    native.native_world_vertices=lambda entry:(np.asarray(entry["original_mesh"].vertices)+actor.positions_m[0],
                                               entry["object"].root_scene_node.absolute_transformation())
    try:
        assert native_plan_checks(v,{},p).passed
        e["motion_episode"]["native_30Hz_readbacks"][450]["body_to_world_m"][0][3]+=.01
        result=native_plan_checks(v,{},p)
        assert any(x.name.endswith("all_root_samples")and x.status=="fail"for x in result.checks)
        e["motion_episode"]["native_30Hz_readbacks"].pop()
        assert any(x.name.endswith("motion_clock")and x.status=="fail"for x in native_plan_checks(v,{},p).checks)
    finally:
        native.native_world_vertices=original


def test_joint_solution_records_proposal_scope_without_claiming_rendered_truth():
    c=context();p=ScenePlan(c.cameras[0],{"a":static_actor("a",[0,0,-4])})
    plugin=SimpleNamespace(candidates=lambda c,r:iter([p]),verify=lambda c,r,p:[CheckResult("branch","pass")])
    r=solve(c,FamilyRequest("QA-21","sound",("a",)),plugin)
    source=r.plan.metadata["joint_solution"]
    assert source["solver_entry"]=="avengine.construction.joint.solve"
    assert source["native_preflight_status"]=="not_run"
    assert not source["native_full_render_and_qa_verified"]


def test_composite_raycaster_preserves_long_registered_identity():
    import trimesh
    from avengine.construction.joint.mesh_prediction import InstanceRaycaster
    wall=trimesh.creation.box((2.,2.,.1));wall.apply_translation([0,0,-4.])
    foreground=trimesh.creation.box((1.,1.,.1));foreground.apply_translation([0,0,-2.])
    identifier="registered_occluder_"+"x"*160
    ray=CompositeRaycaster(MeshRaycaster(wall),InstanceRaycaster(MeshInstance(identifier,MeshRaycaster(foreground))))
    distance,ids=ray.first_hit([0,0,0],[[0,0,-1]])
    assert distance[0]==pytest.approx(1.95,abs=1e-5)
    assert ids[0]==identifier


def test_exact_mesh_roi_matches_full_depth_counts_and_does_not_mutate_background():
    import trimesh
    c=CameraPose((0.,1.25,0.),0.,width=160,height=90)
    mesh=trimesh.creation.box((.6,1.8,.2));transform=np.eye(4);transform[:3,3]=[.3,.9,-4.]
    target=MeshInstance("t",MeshRaycaster(mesh),transform)
    background=depth(c);before=background.depth_m.copy()
    blocker=trimesh.creation.box((.3,2.,.2));blocker.apply_translation([.3,1.,-2.])
    other=MeshInstance("b",MeshRaycaster(blocker))
    result=predict_mesh_visibility(c,target,scene_depth=background,occluders=(other,))
    td=raster_mesh_depth(c,target);od=raster_mesh_depth(c,other)
    mask=np.isfinite(td);visible=mask&(td<=od+1e-5)
    assert result.evidence["target_only_pixels"]==int(mask.sum())
    assert result.visible_pixels_estimate==int(visible.sum())
    np.testing.assert_array_equal(background.depth_m,before)
    x0,y0,x1,y1=result.evidence["raster_roi_xyxy"]
    assert (x1-x0)*(y1-y0)<c.width*c.height/2


def fallback_context():
    import trimesh
    c=context();c.allow_native_map_fallback=True
    for a in c.walkable_map.layers.values():a[:]=False
    c.pathfinder=SimpleNamespace(snap_point=lambda p:np.asarray(p)+[0,.17,0],
                                 distance_to_closest_obstacle=lambda p,r:1.)
    floor=trimesh.creation.box((20.,.1,20.));floor.apply_translation([0,-.05,0])
    c.floor_raycaster=MeshRaycaster(floor)
    return c


def test_map_miss_stays_pending_after_real_floor_navmesh_screen():
    c=fallback_context();track=np.array([[0.,0.,-3.],[0.,0.,-4.]])
    q,report=validate_track(c,track)
    assert report["native_map_fallback"]["status"]=="pending_native_sweep"
    assert report["native_map_fallback"]["continuous_samples"]>=40
    np.testing.assert_array_equal(q[:,1],[0,0])
    p=ScenePlan(c.cameras[0],{"a":static_actor("a",[0,0,-4])})
    assert not certify_plan(c,p).passed
    plugin=SimpleNamespace(candidates=lambda c,r:iter([p]),verify=lambda c,r,p:[CheckResult("branch","pass")])
    assert solve(c,FamilyRequest("QA-21","x",("a",)),plugin).status=="unavailable"
    assert not c.walkable_map.layers["human"].any()


def test_map_fallback_cannot_skip_physical_floor_or_body_clearance():
    c=fallback_context()
    c.pathfinder.distance_to_closest_obstacle=lambda p,r:.2
    with pytest.raises(ConstructionError,match="body_clearance"):
        validate_track(c,[[0,0,-4]])
    c.pathfinder.distance_to_closest_obstacle=lambda p,r:1.
    with pytest.raises(ConstructionError,match="physical_floor"):
        validate_track(c,[[0,.05,-4]])


def test_native_fallback_requires_same_pose_physics_proof_and_live_payload():
    c=fallback_context();p=ScenePlan(c.cameras[0],{"a":static_actor("a",[0,0,-4])})
    plugin=SimpleNamespace(candidates=lambda c,r:iter([p]),verify=lambda c,r,p:[CheckResult("branch","pass")])
    request=FamilyRequest("QA-21","x",("a",),{"require_native_preflight":True})
    proof=[CheckResult(n,"pass")for n in ["native:a:full_pose_physics","native:a:exact_pose","native:identities","native:camera_unchanged"]]
    result=solve(c,request,plugin,prepare_and_verify=lambda *a:PreparedCandidate(object(),tuple(proof)))
    assert result.status=="certified"
    assert any(x.reason=="original_native_full_pose_overrides_coarse_map_miss"for x in result.certificate.checks)
    assert solve(c,request,plugin,prepare_and_verify=lambda *a:PreparedCandidate(object(),tuple(proof[:-1]))).status=="unavailable"
    assert solve(c,request,plugin,prepare_and_verify=lambda *a:PreparedCandidate(None,tuple(proof))).status=="unavailable"


@pytest.mark.parametrize("side",["left","right"])
def test_shared_entry_seed_preserves_side_and_fixed_view(side):
    c=CameraPose((0.,1.25,0.),0.)
    seed=tangent_entry_anchors(c.position_m,[0,0,-4.2],side)
    camera=CameraPose(c.position_m,seed["camera_yaw_deg"])
    body=BodySpec("human",.4,1.9,0.)
    assert framing(camera,seed["start_m"],body)[0]=="outside"
    assert framing(camera,seed["endpoint_m"],body)[0]=="inside"
    assert not seed["native_pixel_truth_verified"]
    assert entry_timing(seed["planning_length_m"],1.05)["start_s"]>.75


def test_floor_height_is_solved_from_real_support_before_native_not_silently_lifted():
    import trimesh
    c=context()
    floor=trimesh.creation.box((10.,.1,10.));floor.apply_translation([0,-.038,0]) # top=.012
    c.floor_raycaster=MeshRaycaster(floor)
    body=trimesh.creation.box((.6,1.8,.4));body.apply_translation([0,.9,0])
    actor=static_actor("a",[0,0,-4],facing_rad=0.)
    variants=list(floor_contact_variants(c,actor,body))
    assert variants and variants[0].positions_m[0,1]==pytest.approx(.012,abs=1e-6)
    np.testing.assert_array_equal(actor.positions_m[:,1],np.zeros(901))
    np.testing.assert_array_equal(variants[0].positions_m[:,[0,2]],actor.positions_m[:,[0,2]])
    assert variants[0].metadata["floor_contact_refinement"]["native_contact_checks_required"]


def test_original_acoustic_gate_receives_actual_primary_and_attached_emitters(tmp_path):
    observed={"emitter_position_m":[1,1.5,2],"native_emitter_present":True,
              "attached_emitter_readbacks":[{"endpoint_id":"a_foot_L","position_m":[1,.03,2]}]}
    entry={"endpoint_id":"a","declaration":{"silent_decor":False}}
    video=SimpleNamespace(actors=[entry],motion_house={"house":"test"},_apply_entry=lambda e,s:observed)
    scene={"events":[{"endpoint_id":"a"},{"endpoint_id":"a_foot_L"}],"listener_m":[0,1.25,0],"camera_forward":[0,0,-1]}
    seen=[];closed=[];runtime={"fresh_rir_workers":5}
    class Gate:
        def __init__(self,s,h,inputs,cache,receipt):
            assert inputs is runtime
        def check(self,ep,p):
            seen.append((ep,p.tolist()))
            return {"status":"blocked"if ep.endswith("L")else"pass","unchanged_gate_deg":5.}
        def close(self):closed.append(True)
    result=native_acoustic_checks(video,scene,runtime,tmp_path,gate_factory=Gate)
    assert seen==[("a",[1.,1.5,2.]),("a_foot_L",[1.,.03,2.])]
    assert closed==[True] and any(c.status=="fail"for c in result)
    assert runtime=={"fresh_rir_workers":5}
    assert scene["joint_acoustic_preflight"]["status"]=="fail"


def test_missing_actual_active_emitter_is_unknown_not_silently_admitted(tmp_path):
    entry={"endpoint_id":"a","declaration":{"silent_decor":True}}
    video=SimpleNamespace(actors=[entry],motion_house={},_apply_entry=lambda e,s:{"native_emitter_present":False})
    scene={"events":[{"endpoint_id":"unbound"}],"listener_m":[0,1,0],"camera_forward":[0,0,-1]}
    class Gate:
        def __init__(self,*a):pass
        def check(self,*a):raise AssertionError("no invented coordinate")
        def close(self):pass
    result=native_acoustic_checks(video,scene,{},tmp_path,gate_factory=Gate)
    assert result[0].status=="unknown"


def test_shared_full_sweep_does_not_skip_a_collision_between_stride_frames(monkeypatch):
    import trimesh
    import avengine.assets.roster_motion.episode as episode
    import avengine.capture.v7_visible_mesh as visible_mesh
    mesh=trimesh.creation.box((.2,.2,.2));calls=[]
    class Verifier:
        def __init__(self,*a):self.minimum_clearance_m=0.;self.record_only=True;self.stride=10
        def check_world(self,world,faces):
            calls.append(True)
            if world[:,0].mean()>.5:raise ValueError("unit collision between stride samples")
            return .02
        def floor_certificate(self):return {"status":"pass","method":"unit fixture"}
    monkeypatch.setattr(episode,"TriangleClearance",Verifier)
    monkeypatch.setattr(visible_mesh,"indexed_static_mesh",lambda p:mesh)
    matrices=np.repeat(np.eye(4)[None],901,axis=0);matrices[899,0,3]=1.
    entry={"declaration":{"asset_id":"a"},"motion_episode":{"physical_ground_y_m":0.,
        "pose_glbs":["unit"]*901,"native_30Hz_readbacks":[{"body_to_world_m":m.tolist()}for m in matrices]}}
    video=SimpleNamespace(actors=[entry],sim=None,motion_inputs={},receipt={},motion_house={})
    result=native_full_mesh_sweep(video)
    assert result[0].status=="fail" and result[0].evidence["failed_frame_30hz"]==899
    assert len(calls)==900


def test_shared_full_sweep_compares_two_movers_at_same_frame(monkeypatch):
    import trimesh,fcl
    import avengine.assets.roster_motion.episode as episode
    import avengine.capture.v7_visible_mesh as visible_mesh
    mesh=trimesh.creation.box((.2,.2,.2))
    class Verifier:
        def __init__(self,video,*a):
            assert len(video.actors)==1 # another mover is not frozen at frame zero
            self.minimum_clearance_m=0.;self.fcl=fcl
        def check_world(self,world,faces):return .02
        def floor_certificate(self):return {"status":"pass","method":"unit fixture"}
    monkeypatch.setattr(episode,"TriangleClearance",Verifier)
    monkeypatch.setattr(visible_mesh,"indexed_static_mesh",lambda p:mesh)
    entries=[]
    for key,sign in [('a',1),('b',-1)]:
        matrices=np.repeat(np.eye(4)[None],901,axis=0);matrices[:,0,3]=sign*np.linspace(-1,1,901)
        entries.append({"declaration":{"asset_id":key},"motion_episode":{"physical_ground_y_m":0.,
            "pose_glbs":["unit"]*901,"native_30Hz_readbacks":[{"body_to_world_m":m.tolist()}for m in matrices]}})
    video=SimpleNamespace(actors=entries,sim=None,motion_inputs={},receipt={},motion_house={})
    result=native_full_mesh_sweep(video)
    assert len(result)==2 and all(c.status=="fail"for c in result)
    assert all(c.reason=="actual_synchronous_mover_collision_or_declared_clearance"for c in result)


def test_extra_circular_body_envelope_can_only_defer_to_actual_native_sweep():
    import trimesh
    c=context();c.allow_native_map_fallback=True
    c.walkable_map.layers["human"][70,94]=False
    c.pathfinder=SimpleNamespace(snap_point=lambda p:np.asarray(p)+[0,.17,0],distance_to_closest_obstacle=lambda p,r:1.)
    floor=trimesh.creation.box((20.,.1,20.));floor.apply_translation([0,-.05,0]);c.floor_raycaster=MeshRaycaster(floor)
    p=ScenePlan(c.cameras[0],{"a":static_actor("a",[0,0,-4],BodySpec("human",.59,1.8))})
    checks=certify_plan(c,p).checks
    assert any(x.name=="a:extra_body_width"and x.status=="unknown"and x.reason=="native_full_pose_required_for_map_fallback"for x in checks)
    c.allow_native_map_fallback=False
    assert any(x.name=="a:extra_body_width"and x.status=="fail"for x in certify_plan(c,p).checks)


def test_fixed_clock_out_and_back_retains_pause_and_original_clock():
    c=context();t=np.arange(901)/30;d=1.02*np.clip(t-1,0,2)+1.02*np.clip(t-6,0,2)
    p=out_and_back(c,[-1,0,-4],[1,0],d,2.04,turn_interval_s=(4.8,5.8))
    np.testing.assert_array_equal(p.metadata["distance_clock_m"],d)
    np.testing.assert_allclose(p.positions_m[-1],p.positions_m[0],atol=1e-8)
    np.testing.assert_allclose(p.positions_m[144:175],np.repeat(p.positions_m[144:145],31,axis=0))
    assert p.headings_rad[-1]-p.headings_rad[0]==pytest.approx(math.pi)
    with pytest.raises(ConstructionError,match="stationary"):
        out_and_back(c,[-1,0,-4],[1,0],d,2.04,turn_interval_s=(2.,2.5))


def test_canonical_turnaround_shape_keeps_clock_and_requires_real_room_validation():
    t=np.arange(901)/30
    d=1.02*np.clip(t-1,0,2)+1.02*np.clip(t-6,0,2)
    p,h=out_and_back_shape([0,0,-4],[1,0],d,2.04,turn_sign=-1)
    np.testing.assert_allclose(np.linalg.norm(np.diff(p,axis=0),axis=1),np.diff(d),atol=1e-12)
    moving=np.diff(d)>1e-8
    directions=np.diff(p[:,[0,2]],axis=0)[moving]/np.diff(d)[moving,None]
    np.testing.assert_allclose(directions,np.column_stack((np.cos(h[:-1][moving]),np.sin(h[:-1][moving]))),atol=1e-12)
    assert h[-1]-h[0]==pytest.approx(-math.pi)
    # A canonical pose cache is valid even where a real room has a wall.
    c=context(holes=[(70,90)])
    with pytest.raises(ConstructionError,match="walkable_map"):
        out_and_back(c,[0,0,-4],[1,0],d,2.04,turn_sign=-1)
    for fps in (0.,-30.,float("nan"),float("inf")):
        with pytest.raises(ConstructionError,match="invalid_out_and_back"):
            out_and_back_shape([0,0,-4],[1,0],d,2.04,fps=fps)


def test_static_domain_mask_applies_before_ranking_without_changing_the_map():
    c=context()
    bodies={"a":BodySpec()}
    original=c.walkable_map.layers["human"].copy()
    plain=next(enumerate_static_layouts(c,bodies,{"a":"inside"},max_candidates=1))
    index=int(np.argmin(np.linalg.norm(c.points_m-np.array([0,0,-6.5]),axis=1)))
    mask=np.zeros(len(c.points_m),bool);mask[index]=True
    assert not np.allclose(plain.actors["a"].positions_m[0,[0,2]],c.points_m[index,[0,2]])
    plans=list(enumerate_static_layouts(c,bodies,{"a":"inside"},point_masks={"a":mask},max_candidates=1))
    assert len(plans)==1
    np.testing.assert_array_equal(plans[0].actors["a"].positions_m[0,[0,2]],c.points_m[index,[0,2]])
    np.testing.assert_array_equal(c.walkable_map.layers["human"],original)
    mask[:]=False
    assert not list(enumerate_static_layouts(c,bodies,{"a":"inside"},point_masks={"a":mask}))
    for bad in (np.ones(len(c.points_m)),np.ones(len(c.points_m)-1,bool)):
        with pytest.raises(ValueError,match="boolean arrays"):
            list(enumerate_static_layouts(c,bodies,{"a":"inside"},point_masks={"a":bad}))


def test_analytic_occluder_cannot_escape_its_explicit_point_domain(monkeypatch):
    c=context()
    c.points_m=np.array([[0.,0.,-6.],[-2.,0.,-4.]])
    monkeypatch.setattr(c,"predict",lambda *a,**kw:SimpleNamespace(state="visible_occluded",occluded_fraction=.5))
    bodies={"target":BodySpec(),"front":BodySpec()}
    kw=dict(roles={"target":"inside","front":"inside"},occlusion_pair=("target","front",.2,.8),max_candidates=1)
    assert list(enumerate_static_layouts(c,bodies,point_masks={"target":np.array([True,False])},**kw))
    assert not list(enumerate_static_layouts(c,bodies,point_masks={
        "target":np.array([True,False]),"front":np.array([False,True])},**kw))


def _parallel_test_actors(c,near_z=-3.5):
    t=np.arange(901)/30
    d=1.2*np.clip(t-1.,0.,1.5)
    e=1.2*np.clip(t-15.,0.,1.5)
    a=bind_distance_clock(c,d,origin_m=[0,0,near_z],heading_rad=-math.pi/2,entity_id="a")
    b=bind_distance_clock(c,e,origin_m=[0,0,near_z-1.8],heading_rad=math.pi/2,entity_id="b")
    return a,b


def test_parallel_lanes_preserve_each_clock_and_filter_blocked_side():
    c=context(holes=[(70,115)])
    a,b=_parallel_test_actors(c)
    original_a=a.positions_m.copy();original_b=b.positions_m.copy()
    stats={}
    plans=list(paired_parallel_routes(c,a,b,camera=c.cameras[0],separations_m=(1.25,),diagnostics=stats))
    assert len(plans)==1 and plans[0].metadata["side"]==-1
    plan=plans[0]
    np.testing.assert_array_equal(plan.actors["a"].positions_m,original_a)
    np.testing.assert_allclose(plan.actors["b"].positions_m-original_b,np.tile([-1.25,0,0],(901,1)),atol=1e-12)
    for actor in (a,b):
        bound=plan.actors[actor.entity_id]
        np.testing.assert_array_equal(bound.headings_rad,actor.headings_rad)
        np.testing.assert_array_equal(bound.metadata["distance_clock_m"],actor.metadata["distance_clock_m"])
    assert plan.metadata["minimum_pair_distance_m"]>=1.25-1e-10
    assert plan.metadata["rest_framing"]["states"]==["inside"]*4
    assert plan.camera.position_m==c.cameras[0].position_m
    assert certify_plan(c,plan).passed and stats["rejections"]
    np.testing.assert_array_equal(a.positions_m,original_a)
    np.testing.assert_array_equal(b.positions_m,original_b)
    assert plan.actors["b"].metadata["fixed_clock_geometry"]["origin_m"][0]==pytest.approx(-1.25)


def test_parallel_lanes_reject_cropped_feet_and_crossing_seed():
    c=context()
    a,b=_parallel_test_actors(c,near_z=-1.2)
    stats={}
    assert not list(paired_parallel_routes(c,a,b,camera=c.cameras[0],separations_m=(1.25,),diagnostics=stats))
    assert stats["rejections"]["parallel_complete_body_framing"]==2
    a,b=_parallel_test_actors(c)
    b.positions_m[:,0]=np.linspace(-1.,1.,901)
    with pytest.raises(ConstructionError,match="shared_straight"):
        list(paired_parallel_routes(c,a,b,camera=c.cameras[0]))
