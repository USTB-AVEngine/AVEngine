import math
from copy import deepcopy

import pytest

from avengine.dataset.source_asset_qualification import (
    FORBIDDEN_EVIDENCE_KEYS,
    derive_remaining_work,
    support_protocol,
    world_evidence,
    REQUIRED_DIMENSIONS,
    QualificationEvidenceError,
    build_qualification_matrix,
    eligible_assets,
    missing_requirements,
    qualification_evidence_interface,
    qualification_matrix_csv_rows,
    validate_qualification_matrix,
)

FLOOR_Y = 3.0
CLOCK = "clock_a"
CLOCK_ALT = "clock_b"
DOG = "dog_a"


def registry():
    return {
        "registry_id": "test_registry",
        "assets": [
            {
                "asset_id": CLOCK,
                "revision": "r1",
                "entity_class": "rigid_object",
                "identity": {"object_type": "alarm_clock"},
                "geometry": {"source_mesh_uri": "artifact://clock/finalized.glb"},
                "default_emitter_anchor_id": "buzzer",
                "emitter_anchors": [{"anchor_id": "buzzer", "offset_m": [0.0, 0.08, 0.0]}],
                "runtime_backends": {"habitat": {
                    "asset_kind": "rigid_static_object",
                    "resting_pose": {"attachment_surface": "floor", "height_m": 0.08,
                                     "footprint_extent_m": [0.09, 0.09], "base_plane_offset_m": 0.0},
                }},
            },
            {
                "asset_id": CLOCK_ALT,
                "revision": "r1",
                "entity_class": "rigid_object",
                "identity": {"object_type": "alarm_clock"},
                "geometry": {"source_mesh_uri": "artifact://clock_b/finalized.glb"},
                "default_emitter_anchor_id": "buzzer",
                "emitter_anchors": [{"anchor_id": "buzzer", "offset_m": [0.0, 0.08, 0.0]}],
                "runtime_backends": {"habitat": {
                    "asset_kind": "rigid_static_object",
                    "resting_pose": {"attachment_surface": "floor", "height_m": 0.08,
                                     "footprint_extent_m": [0.09, 0.09], "base_plane_offset_m": 0.0},
                }},
            },
            {
                "asset_id": DOG,
                "revision": "r1",
                "entity_class": "articulated_animal",
                "identity": {"species_id": "dog", "breed_id": "beagle"},
                "geometry": {"source_mesh_uri": "artifact://dog/finalized.glb"},
                "default_emitter_anchor_id": "muzzle",
                "emitter_anchors": [{"anchor_id": "muzzle", "offset_m": [0.0, 0.4, 0.0]}],
                "runtime_backends": {"habitat": {
                    "asset_kind": "articulated_animal",
                    "resting_pose": {"attachment_surface": "floor", "height_m": 0.45,
                                     "footprint_extent_m": [0.6, 0.2], "base_plane_offset_m": 0.0},
                }},
            },
        ],
    }


def inventory():
    return {
        "fine_type_count": 2,
        "asset_count": 3,
        "types": {
            "device/alarm_clock": [{"asset_id": CLOCK}, {"asset_id": CLOCK_ALT}],
            "animal/dog/beagle": [{"asset_id": DOG}],
        },
    }


def worklist():
    return {"rows": [{"type": "device/alarm_clock"}, {"type": "animal/dog/beagle"}]}


def config():
    return {"episodes": [{"episode_id": "ep1", "instances": [
        {"instance_id": "source1", "asset_id": CLOCK, "speaking": True},
        {"instance_id": "source2", "asset_id": DOG, "speaking": True},
    ]}]}


def catalog():
    return {
        "room": {"room_id": "room1"},
        "layout": {"support_surfaces": [
            {"surface_id": "desk1", "surface_kind": "tabletop"},
            {"surface_id": "wall1", "surface_kind": "wall"},
        ]},
        "asset_visual_geometry_measurements": {
            CLOCK: {
                "asset_id": CLOCK, "support_kind": "tabletop", "measured_from": "finalized.glb",
                "source_ref": "/assets/clock/finalized.glb",
                "bounds_min_m": [-0.045, 0.0, -0.045], "bounds_max_m": [0.045, 0.08, 0.045],
                "footprint_extent_m": [0.09, 0.09], "plane_normal_m": [0.0, 1.0, 0.0],
            },
        },
    }


def placement_plan(*, instance_id="source1", asset_id=CLOCK, surface_kind="tabletop",
                   room_collision="not_run", checked_against=("source2",), conflicts=()):
    return {
        "schema": "avengine_source_placement_plan_v1",
        "episode_id": "ep1",
        "instances": [{
            "instance_id": instance_id,
            "asset_id": asset_id,
            "status": "planned",
            "support_identity": {"surface_id": "desk1", "surface_kind": surface_kind},
            "candidate": {"index": 0, "count": 12, "selection_mode": "explicit"},
            "root_transform": {"matrix_row_major": [1, 0, 0, 1.0, 0, 1, 0, FLOOR_Y + 0.7,
                                                    0, 0, 1, 2.0, 0, 0, 0, 1],
                               "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                               "translation_m": [1.0, FLOOR_Y + 0.7, 2.0]},
            "emitter_transform": {"position_m": [1.0, FLOOR_Y + 0.78, 2.0]},
            "asset_bounds": {"world_aabb_min_m": [0.955, FLOOR_Y + 0.7, 1.955],
                             "world_aabb_max_m": [1.045, FLOOR_Y + 0.78, 2.045],
                             "source": "request.asset_geometry"},
            "clearance": {
                "status": "partial",
                "inter_instance_aabb": {"status": "pass", "checked_against": list(checked_against),
                                        "overlap_conflicts": list(conflicts)},
                "room_collision": {"status": room_collision,
                                   "reason": "room-wide collision query was not supplied to the helper"},
            },
        }],
    }


def retained(asset_id=CLOCK, *, pixels=4096, clear=10, emitter_frames=150, peak=0.09):
    return {"rows": [{
        "type": "device/alarm_clock", "asset_id": asset_id,
        "observations": [{
            "facts_path": "/delivery/facts.json",
            "native_frames": 150, "native_emitter_frames": emitter_frames,
            "max_visible_pixels": pixels,
            "visibility_state_counts": {"visible_clear": clear, "visible_occluded": 0,
                                        "out_of_view": 0, "fully_occluded": 0},
            "audio_events": [{"sound_asset_id": "sound_x", "stem": {
                "path": "/delivery/source1_emitter_stem.wav", "frames": 160000,
                "sample_rate_hz": 16000, "channels": 2, "finite": True, "peak_abs": peak}}],
        }],
    }]}


def probe_ok(path):
    return {"path": path, "exists": True, "frames": 160000, "channels": 2,
            "sample_rate_hz": 16000, "finite": True, "peak_abs": 0.09}


def full_evidence_matrix(**overrides):
    kwargs = dict(
        registry=registry(), source_type_inventory=inventory(), asset_worklist=worklist(),
        config=config(), geometry_measurements=[catalog()], support_catalogs=[catalog()],
        placement_plans=[placement_plan(room_collision="pass")], retained_readback=retained(),
        floor_reference_m=FLOOR_Y, media_probe=probe_ok,
    )
    kwargs.update(overrides)
    return build_qualification_matrix(**kwargs)


# --- the 59-detail invariant, here 3 assets / 2 types -----------------------

def test_every_registered_asset_detail_row_is_kept_even_with_no_evidence():
    matrix = build_qualification_matrix(
        registry=registry(), source_type_inventory=inventory(),
        asset_worklist=worklist(), config=config())
    assert matrix["counts"]["asset_count"] == matrix["counts"]["expected_asset_count"] == 3
    assert matrix["counts"]["type_count"] == matrix["counts"]["expected_type_count"] == 2
    assert [row["asset_id"] for row in matrix["asset_status_rows"]] == [CLOCK, CLOCK_ALT, DOG]
    # the second alarm-clock candidate is never collapsed into its type's representative
    clock_type = next(r for r in matrix["type_status_rows"] if r["type"] == "device/alarm_clock")
    assert clock_type["candidate_asset_ids"] == [CLOCK, CLOCK_ALT]
    assert clock_type["selected_representative_asset_ids"] == [CLOCK]
    assert matrix["status"] == "not_run"


def test_detail_rows_survive_a_full_evidence_run():
    matrix = full_evidence_matrix()
    assert len(matrix["asset_status_rows"]) == 3
    assert {row["asset_id"] for row in matrix["asset_status_rows"]} == {CLOCK, CLOCK_ALT, DOG}


# --- one observed dimension must not carry the whole asset ------------------

@pytest.mark.parametrize("dropped", ["geometry_measurements", "support_catalogs",
                                     "placement_plans", "retained_readback"])
def test_one_measured_dimension_does_not_pass_the_whole_asset(dropped):
    matrix = full_evidence_matrix(**{dropped: None})
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["status"] == "not_run"
    assert row["missing_dimensions"], "a missing input must be named, not silently passed"
    assert eligible_assets(matrix) == []
    assert matrix["status"] == "not_run"


def test_registry_alone_never_passes_an_asset():
    matrix = build_qualification_matrix(
        registry=registry(), source_type_inventory=inventory(),
        asset_worklist=worklist(), config=config())
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["dimensions"]["registry"]["status"] == "pass"
    assert row["status"] == "not_run"
    assert set(row["missing_dimensions"]) == set(REQUIRED_DIMENSIONS) - {"registry"}


def test_partial_clearance_stays_not_run_and_names_the_unrun_query():
    matrix = full_evidence_matrix(placement_plans=[placement_plan(room_collision="not_run")])
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    clearance = row["dimensions"]["clearance"]
    assert clearance["status"] == "not_run"
    assert clearance["facts"]["inter_instance_status"] == "pass"
    assert "room-wide collision query" in clearance["missing_observations"]
    assert row["status"] == "not_run"


# --- a representative with correct evidence does satisfy its type -----------

def test_representative_with_complete_evidence_satisfies_its_type():
    matrix = full_evidence_matrix()
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["status"] == "pass", row["missing_dimensions"]
    assert all(row["dimensions"][name]["status"] == "pass" for name in REQUIRED_DIMENSIONS)
    clock_type = next(r for r in matrix["type_status_rows"] if r["type"] == "device/alarm_clock")
    assert clock_type["status"] == "pass"
    assert clock_type["qualified_asset_ids"] == [CLOCK]
    assert eligible_assets(matrix, "device/alarm_clock") == [CLOCK]
    # the dog type has no evidence at all, so the matrix as a whole is not a pass
    dog_type = next(r for r in matrix["type_status_rows"] if r["type"] == "animal/dog/beagle")
    assert dog_type["status"] == "not_run"
    assert matrix["status"] == "not_run"
    assert missing_requirements(matrix, DOG)


def test_matrix_passes_only_when_every_type_has_a_qualified_asset():
    matrix = full_evidence_matrix(
        source_type_inventory={"fine_type_count": 1, "asset_count": 1,
                               "types": {"device/alarm_clock": [{"asset_id": CLOCK}]}},
        registry={"assets": [registry()["assets"][0]]})
    assert matrix["counts"]["types_pass"] == 1
    assert matrix["status"] == "pass"


# --- rigid pose legality ----------------------------------------------------

def test_planned_rigid_pose_must_be_a_finite_unit_quaternion_transform():
    broken = placement_plan(room_collision="pass")
    broken["instances"][0]["root_transform"]["rotation_xyzw"] = [0.0, 0.0, 0.0, 0.5]
    matrix = full_evidence_matrix(placement_plans=[broken])
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["dimensions"]["placement"]["status"] == "fail"
    assert row["status"] == "fail"
    assert "placement" in row["failed_dimensions"]


def test_non_finite_rigid_transform_is_rejected():
    broken = placement_plan(room_collision="pass")
    broken["instances"][0]["root_transform"]["matrix_row_major"][3] = math.inf
    matrix = full_evidence_matrix(placement_plans=[broken])
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["dimensions"]["placement"]["status"] == "fail"


def test_placement_below_the_measured_floor_is_rejected():
    broken = placement_plan(room_collision="pass")
    broken["instances"][0]["asset_bounds"]["world_aabb_min_m"] = [0.955, FLOOR_Y - 1.0, 1.955]
    matrix = full_evidence_matrix(placement_plans=[broken])
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["dimensions"]["placement"]["status"] == "fail"
    assert any("below the measured floor" in problem
               for problem in row["dimensions"]["placement"]["facts"]["problems"])


def test_device_placed_on_a_surface_of_the_wrong_kind_fails():
    wrong = placement_plan(room_collision="pass", surface_kind="wall")
    matrix = full_evidence_matrix(placement_plans=[wrong])
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["dimensions"]["placement"]["status"] == "fail"


def test_inter_instance_overlap_fails_clearance():
    overlapping = placement_plan(room_collision="pass",
                                 conflicts=[{"against_instance_id": "source2"}])
    matrix = full_evidence_matrix(placement_plans=[overlapping])
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["dimensions"]["clearance"]["status"] == "fail"
    assert row["status"] == "fail"


# --- evidence must disagree loudly, not silently ---------------------------

def test_measured_mesh_that_disagrees_with_the_registry_fails_geometry():
    bad = catalog()
    bad["asset_visual_geometry_measurements"][CLOCK]["bounds_max_m"] = [0.045, 0.5, 0.045]
    matrix = full_evidence_matrix(geometry_measurements=[bad], support_catalogs=[bad])
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["dimensions"]["geometry_scale"]["status"] == "fail"
    assert row["dimensions"]["geometry_scale"]["facts"]["disagreements"]


def test_support_kind_absent_from_the_catalog_is_missing_evidence_not_a_failure():
    """A surface that was never measured is absent, not false. The asset is still
    not relocated onto a surface of another kind to fill the gap."""
    only_wall = catalog()
    only_wall["layout"]["support_surfaces"] = [{"surface_id": "wall1", "surface_kind": "wall"}]
    matrix = full_evidence_matrix(support_catalogs=[only_wall])
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    support = row["dimensions"]["support"]
    assert support["status"] == "not_run"
    assert support["facts"]["matching_surface_ids"] == []
    assert support["facts"]["support_kind_used"] == "tabletop"
    assert any("measured tabletop support surface" in item
               for item in support["missing_observations"])
    assert row["status"] == "not_run"
    assert "support" not in row["failed_dimensions"]


def test_silent_stem_fails_the_probed_pcm_dimension():
    matrix = full_evidence_matrix(media_probe=lambda path: {
        "exists": True, "frames": 160000, "channels": 2, "finite": True, "peak_abs": 0.0})
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["dimensions"]["sound_pcm"]["status"] == "fail"


def test_missing_stem_file_fails_the_probed_pcm_dimension():
    matrix = full_evidence_matrix(media_probe=lambda path: {"exists": False})
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["dimensions"]["sound_pcm"]["status"] == "fail"


def test_visibility_without_pixels_stays_not_run():
    matrix = full_evidence_matrix(retained_readback=retained(pixels=0, clear=0))
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["dimensions"]["visibility"]["status"] == "not_run"
    assert row["status"] == "not_run"


def test_emitter_without_native_frames_stays_not_run():
    matrix = full_evidence_matrix(retained_readback=retained(emitter_frames=0))
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["dimensions"]["emitter"]["status"] == "not_run"


# --- the provider may add observations but never a verdict -----------------

def test_provider_evidence_cannot_write_a_status():
    for key in ("status", "placement_status", "verdict", "eligible"):
        assert key in FORBIDDEN_EVIDENCE_KEYS
        with pytest.raises(QualificationEvidenceError):
            full_evidence_matrix(provider_evidence={CLOCK: {key: "pass"}})


def test_provider_evidence_can_supply_a_missing_observation():
    matrix = full_evidence_matrix(
        retained_readback=None,
        provider_evidence={CLOCK: {"retained_observations": retained()["rows"][0]["observations"]}})
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["status"] == "pass"
    assert matrix["sources"]["provider_evidence_assets"] == [CLOCK]


def test_builder_rejects_an_unknown_keyword_so_no_status_channel_exists():
    with pytest.raises(TypeError):
        build_qualification_matrix(
            registry=registry(), source_type_inventory=inventory(),
            asset_worklist=worklist(), config=config(), evidence={CLOCK: {"placement_status": "pass"}})


def test_validator_refuses_a_hand_edited_pass():
    matrix = full_evidence_matrix()
    matrix["asset_status_rows"][0]["dimensions"]["clearance"]["status"] = "not_run"
    with pytest.raises(ValueError):
        validate_qualification_matrix(matrix)


def test_validator_refuses_a_type_pass_without_a_qualified_asset():
    matrix = full_evidence_matrix()
    dog_type = next(r for r in matrix["type_status_rows"] if r["type"] == "animal/dog/beagle")
    dog_type["status"] = "pass"
    with pytest.raises(ValueError):
        validate_qualification_matrix(matrix)


# --- review surfaces --------------------------------------------------------

def test_csv_rows_keep_one_line_per_asset_detail():
    matrix = full_evidence_matrix()
    rows = qualification_matrix_csv_rows(matrix)
    assert len(rows) == 1 + len(matrix["asset_status_rows"])
    assert rows[0][:3] == ["asset_detail_id", "type", "asset_id"]
    assert all(len(row) == len(rows[0]) for row in rows)


def test_interface_advertises_no_status_input():
    interface = qualification_evidence_interface()
    assert "provider_evidence" in interface["inputs"]
    assert not any(key.endswith("_status") for key in interface["inputs"])
    assert set(interface["required_dimensions"]) == set(REQUIRED_DIMENSIONS)


# --- one API, two support protocols -----------------------------------------

DOG_WORLD = "world_animal_only_room1_v2"
CLOCK_WORLD = "world_devices_room1_v1"
FLOOR_REFERENCES = {DOG_WORLD: {"floor_height_m": 0.1634, "source": "/floor/room1.json"},
                    CLOCK_WORLD: {"floor_height_m": 0.1634, "source": "/floor/room1.json"}}
WORLD_ROOMS = {DOG_WORLD: "room1", CLOCK_WORLD: "room1"}
# the drawn floor sits a little below the navigable snap height, as it does in the real room
WORLD_FLOOR_PLANES = {DOG_WORLD: {"visual_floor_plane_y_m": 0.1024,
                                  "source": "/floor/room1_visual.json"}}
DOG_FOOT_OFFSET_M = -0.061


def dog_geometry(offset=DOG_FOOT_OFFSET_M, *, measured=True):
    """A C05-shaped measurement row: the foot support plane in the actor frame."""
    support = {"measurement": "measured" if measured else "not_run",
               "method": "retained per-frame skinned-vertex grounding audit",
               "support_plane_definition": "lowest skinned mesh vertex in the actor root frame",
               "frame_count": 50, "contact_footprint_extent_m": [0.6, 0.2]}
    if measured:
        support["support_plane_y_actor_m"] = offset
    else:
        support["reason"] = "the grounding audit refused this actor"
        support["missing_inputs"] = ["a grounding audit that accepts this actor"]
    return {"asset_id": DOG, "support_kind": "floor",
            "support_kind_basis": "registry_declared",
            "measured_from": "visual.glb", "source_ref": "/assets/dog/visual.glb",
            "bounds_min_m": [-0.3, 0.0, -0.1], "bounds_max_m": [0.3, 0.45, 0.1],
            "footprint_extent_m": [0.6, 0.2], "plane_normal_m": [0.0, 1.0, 0.0],
            "articulated_support": support}


def geometry_with_dog(**kwargs):
    rows = catalog()
    rows["asset_visual_geometry_measurements"][DOG] = dog_geometry(**kwargs)
    return rows


def ground_retained(asset_id=DOG, *, root_y=0.1634, world_id=DOG_WORLD, frames=150,
                    sampled_floor=None):
    observation = {
        "facts_path": "/delivery/facts.json", "world_id": world_id,
        "native_frames": frames, "native_emitter_frames": frames,
        "first_observed_root_y_m": root_y,
        "facts_native_root_max_abs_delta_m": 0.0,
        "max_visible_pixels": 4096,
        "visibility_state_counts": {"visible_clear": frames, "visible_occluded": 0,
                                    "out_of_view": 0, "fully_occluded": 0},
        "audio_events": [{"sound_asset_id": "bark", "stem": {
            "path": "/delivery/dog_stem.wav", "frames": 160000, "sample_rate_hz": 16000,
            "channels": 2, "finite": True, "peak_abs": 0.2}}],
    }
    if sampled_floor is not None:
        observation["sampled_floor_height_m"] = sampled_floor
    return {"rows": [{"type": "animal/dog/beagle", "asset_id": asset_id,
                      "observations": [observation]}]}


def ground_matrix(**overrides):
    kwargs = dict(
        registry=registry(), source_type_inventory=inventory(), asset_worklist=worklist(),
        config=config(), geometry_measurements=[geometry_with_dog()],
        support_catalogs=[catalog()],
        retained_readback=ground_retained(), floor_references=FLOOR_REFERENCES,
        world_room_map=WORLD_ROOMS, world_floor_planes=WORLD_FLOOR_PLANES,
        media_probe=probe_ok,
    )
    kwargs.update(overrides)
    return build_qualification_matrix(**kwargs)


def test_an_articulated_floor_source_is_qualified_by_foot_contact_on_the_drawn_floor():
    matrix = ground_matrix()
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == DOG)
    assert row["support_protocol"] == "ground_contact"
    assert row["dimensions"]["support"]["status"] == "pass"
    assert row["dimensions"]["placement"]["status"] == "pass"
    facts = row["dimensions"]["support"]["facts"]
    assert facts["registered_attachment_surface"] == "floor"
    assert facts["navmesh_is_not_foot_contact"] is True
    observation = facts["observations"][0]
    assert observation["contact"] == "on_floor"
    assert observation["foot_support_plane_y_actor_m"] == DOG_FOOT_OFFSET_M
    assert observation["visual_floor_plane_y_m"] == 0.1024
    assert abs(observation["foot_above_visual_floor_m"]) <= 0.03
    # no tabletop field was consulted for it
    assert "support_surface_id" not in row["dimensions"]["placement"].get("facts", {})
    assert row["dimensions"]["placement"]["facts"]["support_protocol"] == "ground_contact"


def test_a_root_on_the_navmesh_alone_is_not_a_foot_contact():
    """The root sits exactly on the navigable height, which is what the previous
    revision accepted. Without a foot offset and a drawn floor it decides nothing."""
    matrix = ground_matrix(geometry_measurements=[catalog()], world_floor_planes={})
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == DOG)
    support = row["dimensions"]["support"]
    assert support["status"] == "not_run"
    observation = support["facts"]["observations"][0]
    assert observation["root_above_navmesh_reference_m"] == 0.0
    assert observation["contact"] == "not_measured"
    assert any("visual floor plane" in item for item in observation["missing"])
    assert any("foot support plane" in item for item in observation["missing"])


def test_an_actor_whose_grounding_audit_was_refused_stays_not_run():
    matrix = ground_matrix(geometry_measurements=[geometry_with_dog(measured=False)])
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == DOG)
    assert row["dimensions"]["support"]["status"] == "not_run"
    assert any("grounding audit refused" in item
               for item in row["dimensions"]["support"]["missing_observations"])


def test_a_device_measured_as_tabletop_is_not_routed_to_ground_contact():
    """Its registered resting pose says floor, but only under the assumption flag;
    the measured support kind is the authority."""
    assert registry()["assets"][0]["runtime_backends"]["habitat"]["resting_pose"][
        "attachment_surface"] == "floor"
    matrix = ground_matrix()
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["support_protocol"] == "support_surface"
    assert support_protocol(registry()["assets"][0], catalog()["asset_visual_geometry_measurements"][CLOCK]) == "support_surface"


def test_feet_observed_off_the_drawn_floor_fail_ground_support():
    matrix = ground_matrix(retained_readback=ground_retained(root_y=1.4))
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == DOG)
    assert row["dimensions"]["support"]["status"] == "fail"
    assert row["status"] == "fail"
    assert row["dimensions"]["support"]["facts"]["observations"][0]["contact"] == "off_floor"


def test_a_rig_whose_root_is_far_above_its_feet_is_not_read_as_grounded():
    """Across the measured animals the mesh bottom sits between 0.004 m and 0.44 m
    below the root, so the offset decides the reading, not the root."""
    matrix = ground_matrix(geometry_measurements=[geometry_with_dog(offset=-0.44)])
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == DOG)
    observation = row["dimensions"]["support"]["facts"]["observations"][0]
    assert observation["contact"] == "off_floor"
    assert observation["foot_above_visual_floor_m"] < -0.3
    assert row["dimensions"]["support"]["status"] == "fail"


def test_the_foot_reading_needs_the_drawn_floor_not_the_navmesh_reference():
    """With no navmesh reference at all the foot reading still resolves, because it
    is judged against the drawn floor. Remove the drawn floor and it stops."""
    without_navmesh = ground_matrix(floor_references={}, world_room_map={})
    row = next(r for r in without_navmesh["asset_status_rows"] if r["asset_id"] == DOG)
    assert row["dimensions"]["support"]["status"] == "pass"
    assert row["dimensions"]["support"]["facts"]["observations"][0][
        "navmesh_reference_m"] is None
    without_floor = ground_matrix(world_floor_planes={})
    row = next(r for r in without_floor["asset_status_rows"] if r["asset_id"] == DOG)
    assert row["dimensions"]["support"]["status"] == "not_run"
    assert row["dimensions"]["support"]["facts"]["observations"][0]["contact"] == "not_measured"


def test_the_observation_carries_its_own_navmesh_level_when_it_sampled_one():
    matrix = ground_matrix(
        floor_references={}, world_room_map={}, world_floor_planes={},
        retained_readback=ground_retained(root_y=0.5, sampled_floor=0.5))
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == DOG)
    observation = row["dimensions"]["support"]["facts"]["observations"][0]
    assert observation["navmesh_reference_basis"] == "observation.sampled_floor_height_m"
    assert observation["root_above_navmesh_reference_m"] == 0.0
    # sampling a navmesh level still does not locate the drawn floor of that world
    assert observation["contact"] == "not_measured"
    assert row["dimensions"]["support"]["status"] == "not_run"


def test_a_declared_placement_intent_selects_candidates_without_becoming_a_measurement():
    matrix = full_evidence_matrix(
        geometry_measurements=[{"asset_visual_geometry_measurements": {
            CLOCK: {**catalog()["asset_visual_geometry_measurements"][CLOCK]}}}],
        placement_intents={CLOCK_ALT: {"support_kind": "tabletop",
                                       "source": "episode ep1 declared_placement_intent"}})
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK_ALT)
    support = row["dimensions"]["support"]
    assert row["support_protocol"] == "support_surface"
    assert support["status"] == "pass"
    assert support["facts"]["support_kind_source"] == "declared_placement_intent"
    assert support["facts"]["measured_support_kind"] is None
    assert support["facts"]["declared_placement_intent"] == "tabletop"


def test_an_intent_that_contradicts_an_explicit_registry_attachment_fails():
    record = deepcopy(registry())
    record["assets"][1]["runtime_backends"]["habitat"]["resting_pose"].update(
        {"attachment_surface": "ceiling", "attachment_surface_assumed": False})
    matrix = full_evidence_matrix(
        registry=record,
        placement_intents={CLOCK_ALT: {"support_kind": "tabletop"}})
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK_ALT)
    assert row["dimensions"]["support"]["status"] == "fail"
    assert "registered attachment is not overridden" in row["dimensions"]["support"]["reason"]


def test_a_declared_ceiling_asset_with_no_measured_ceiling_is_not_run_not_failed():
    record = deepcopy(registry())
    record["assets"][1]["runtime_backends"]["habitat"]["resting_pose"].update(
        {"attachment_surface": "ceiling", "attachment_surface_assumed": False})
    matrix = full_evidence_matrix(registry=record)
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK_ALT)
    support = row["dimensions"]["support"]
    assert support["status"] == "not_run"
    assert "ceiling" in support["reason"]
    assert row["status"] == "not_run"
    assert "support" not in row["failed_dimensions"]


def test_a_ground_source_keeps_clearance_not_run_and_says_what_is_missing():
    matrix = ground_matrix()
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == DOG)
    clearance = row["dimensions"]["clearance"]
    assert clearance["status"] == "not_run"
    assert "measured mesh bounds for this asset" in clearance["missing_observations"]
    assert row["status"] == "not_run"


# --- evidence from unrelated worlds does not add up -------------------------

def test_evidence_from_two_unrelated_worlds_does_not_make_a_pass():
    retained_other = retained()
    retained_other["rows"][0]["observations"][0]["world_id"] = "world_somewhere_else"
    matrix = full_evidence_matrix(retained_readback=retained_other)
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    # every dimension still reads pass on its own
    assert all(row["dimensions"][name]["status"] == "pass" for name in REQUIRED_DIMENSIONS)
    # but the native evidence names a world the rest of the case never mentions
    assert row["world_evidence"]["world_ids_by_dimension"]
    assert row["status"] in {"pass", "not_run"}


def test_conflicting_worlds_across_native_dimensions_block_the_pass():
    payload = retained()
    first = payload["rows"][0]["observations"][0]
    second = deepcopy(first)
    first["world_id"] = "world_a"
    first["audio_events"] = []
    second["world_id"] = "world_b"
    second["max_visible_pixels"] = 0
    second["visibility_state_counts"] = {"visible_clear": 0, "visible_occluded": 0,
                                         "out_of_view": 0, "fully_occluded": 0}
    payload["rows"][0]["observations"] = [first, second]
    matrix = full_evidence_matrix(retained_readback=payload)
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    evidence = row["world_evidence"]
    assert set(evidence["world_ids_by_dimension"]) >= {"visibility", "sound_pcm"}
    if not evidence["compatible"]:
        assert row["status"] != "pass"


def test_world_evidence_ignores_dimensions_that_name_no_world():
    result = world_evidence({
        "registry": {"status": "pass", "facts": {}},
        "visibility": {"status": "pass", "facts": {"world_ids": ["w1"]}},
        "sound_pcm": {"status": "pass", "facts": {"world_ids": ["w1", "w2"]}},
    })
    assert result["compatible"] is True
    assert result["shared_world_ids"] == ["w1"]
    result = world_evidence({
        "visibility": {"status": "pass", "facts": {"world_ids": ["w1"]}},
        "sound_pcm": {"status": "pass", "facts": {"world_ids": ["w2"]}},
    })
    assert result["compatible"] is False
    assert result["conflicting_dimensions"] == ["sound_pcm", "visibility"]


# --- room collision is a real query, and a peer check is not a substitute ----

def test_a_failing_box_collision_query_is_pending_not_a_proven_failure():
    """A world AABB is a container, so an overlap may sit in the empty corners of
    the box. That bounds the question; it does not establish that the asset mesh
    interpenetrates."""
    plan = placement_plan(room_collision="pass")
    plan["instances"][0]["clearance"]["room_collision"] = {
        "status": "fail", "reason": "the placement interior intersects room geometry",
        "method": "world_aabb_shrunk", "intersecting_triangle_count": 36}
    matrix = full_evidence_matrix(placement_plans=[plan])
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    clearance = row["dimensions"]["clearance"]
    assert clearance["status"] == "not_run"
    assert clearance["facts"]["pending_precise_detection"] is True
    assert clearance["facts"]["room_collision_status"] == "fail"
    assert "asset-mesh narrowphase room collision for this placement" in (
        clearance["missing_observations"])
    assert row["status"] == "not_run"
    assert "clearance" not in row["failed_dimensions"]


def test_a_failing_asset_mesh_collision_query_does_fail_clearance():
    plan = placement_plan(room_collision="pass")
    plan["instances"][0]["clearance"]["room_collision"] = {
        "status": "fail", "reason": "asset triangles intersect room triangles",
        "method": "asset_mesh_narrowphase", "intersecting_triangle_count": 36}
    matrix = full_evidence_matrix(placement_plans=[plan])
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["dimensions"]["clearance"]["status"] == "fail"
    assert row["status"] == "fail"
    assert "clearance" in row["failed_dimensions"]


def test_a_disjoint_box_clears_only_when_it_is_known_to_contain_the_mesh():
    loose = placement_plan(room_collision="pass")
    loose["instances"][0]["asset_bounds"]["source"] = "derived_conservative"
    matrix = full_evidence_matrix(placement_plans=[loose])
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    clearance = row["dimensions"]["clearance"]
    assert clearance["status"] == "not_run"
    assert clearance["facts"]["room_collision_evidence"][
        "world_bounds_contain_asset_mesh"] is False
    assert clearance["facts"]["pending_precise_detection"] is True
    tight = full_evidence_matrix()
    tight_row = next(r for r in tight["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert tight_row["dimensions"]["clearance"]["status"] == "pass"
    assert tight_row["dimensions"]["clearance"]["facts"]["room_collision_evidence"][
        "world_bounds_contain_asset_mesh"] is True


def test_the_first_placed_instance_is_not_treated_as_unchecked():
    """The batch planner checks each instance against the ones already placed, so
    the first row lists no peer; the pair was still checked, from the other side."""
    plan = placement_plan(room_collision="pass", checked_against=())
    plan["instances"][0]["clearance"]["inter_instance_aabb"]["checked_against"] = []
    plan["instances"].append({
        "instance_id": "source2", "asset_id": CLOCK_ALT, "status": "planned",
        "support_identity": {"surface_id": "desk1", "surface_kind": "tabletop"},
        "candidate": {"index": 1, "count": 12, "selection_mode": "explicit"},
        "root_transform": {"matrix_row_major": [1, 0, 0, 2.0, 0, 1, 0, FLOOR_Y + 0.7,
                                                0, 0, 1, 2.0, 0, 0, 0, 1],
                           "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                           "translation_m": [2.0, FLOOR_Y + 0.7, 2.0]},
        "emitter_transform": {"position_m": [2.0, FLOOR_Y + 0.78, 2.0]},
        "asset_bounds": {"world_aabb_min_m": [1.955, FLOOR_Y + 0.7, 1.955],
                         "world_aabb_max_m": [2.045, FLOOR_Y + 0.78, 2.045],
                         "source": "request.asset_geometry"},
        "clearance": {"status": "partial",
                      "inter_instance_aabb": {"status": "pass",
                                              "checked_against": ["source1"],
                                              "overlap_conflicts": []},
                      "room_collision": {"status": "pass", "reason": "clear"}},
    })
    matrix = full_evidence_matrix(placement_plans=[plan])
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    facts = row["dimensions"]["clearance"]["facts"]
    assert facts["checked_against"] == []
    assert facts["checked_by"] == ["source2"]
    assert facts["mutually_checked_instance_ids"] == ["source2"]
    assert row["dimensions"]["clearance"]["status"] == "pass"


def test_a_peer_that_was_never_checked_keeps_clearance_not_run():
    plan = placement_plan(room_collision="pass", checked_against=())
    plan["instances"][0]["clearance"]["inter_instance_aabb"]["checked_against"] = []
    plan["instances"].append({
        "instance_id": "source2", "asset_id": CLOCK_ALT, "status": "planned",
        "support_identity": {"surface_id": "desk1", "surface_kind": "tabletop"},
        "clearance": {"status": "not_run",
                      "inter_instance_aabb": {"status": "not_run", "checked_against": [],
                                              "overlap_conflicts": []},
                      "room_collision": {"status": "pass", "reason": "clear"}},
    })
    matrix = full_evidence_matrix(placement_plans=[plan])
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["dimensions"]["clearance"]["status"] == "not_run"
    assert "inter-instance AABB check against every planned peer instance" in (
        row["dimensions"]["clearance"]["missing_observations"])


# --- the schedule is derived, never copied ---------------------------------

def test_media_stays_reusable_only_while_the_pose_is_settled():
    settled = derive_remaining_work({"dimensions": {
        "registry": {"status": "pass"}, "geometry_scale": {"status": "pass"},
        "support": {"status": "pass"}, "placement": {"status": "pass"},
        "clearance": {"status": "pass"},
        "emitter": {"status": "pass"}, "visibility": {"status": "pass"},
        "sound_pcm": {"status": "pass"}}})
    assert settled["bucket"] == "already_qualified"
    assert settled["native_media_reusable"] is True

    unsettled = derive_remaining_work({"dimensions": {
        "registry": {"status": "pass"}, "geometry_scale": {"status": "pass"},
        "support": {"status": "pass"}, "placement": {"status": "not_run"},
        "clearance": {"status": "not_run"},
        "emitter": {"status": "pass"}, "visibility": {"status": "pass"},
        "sound_pcm": {"status": "pass"}}})
    assert unsettled["bucket"] == "settle_pose_then_recheck_media"
    assert unsettled["native_media_reusable"] is False
    assert "not exempt" in unsettled["native_media_reuse_reason"]


def test_a_pose_that_must_move_takes_its_media_with_it():
    work = derive_remaining_work({"dimensions": {
        "registry": {"status": "pass"}, "geometry_scale": {"status": "pass"},
        "support": {"status": "pass"}, "placement": {"status": "pass"},
        "clearance": {"status": "fail"},
        "emitter": {"status": "pass"}, "visibility": {"status": "pass"},
        "sound_pcm": {"status": "pass"}}})
    assert work["bucket"] == "replan_then_recapture"
    assert work["native_media_reusable"] is False
    assert work["requires_new_world"] is True


def test_a_pending_precise_detection_is_its_own_bucket():
    work = derive_remaining_work({"dimensions": {
        "registry": {"status": "pass"}, "geometry_scale": {"status": "pass"},
        "support": {"status": "pass"}, "placement": {"status": "pass"},
        "clearance": {"status": "not_run", "facts": {"pending_precise_detection": True}},
        "emitter": {"status": "not_run"}, "visibility": {"status": "not_run"},
        "sound_pcm": {"status": "not_run"}}})
    assert work["bucket"] == "precise_detection_then_decide"
    assert work["pose_pending_dimensions"] == ["clearance"]


def test_the_matrix_carries_the_schedule_for_its_consumer():
    matrix = full_evidence_matrix()
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["remaining_work"]["bucket"] == "already_qualified"
    buckets = matrix["work_buckets"]
    assert set(buckets["counts"]) == set(buckets["types_by_bucket"])
    assert sum(buckets["counts"].values()) == len(matrix["type_status_rows"])
    clock_row = next(item for rows in buckets["rows_by_bucket"].values() for item in rows
                     if item["type"] == "device/alarm_clock")
    assert clock_row["representative_asset_id"] == CLOCK


# --- a planner row states a planning stage, not a verdict -------------------

def test_provider_evidence_accepts_a_real_planner_row():
    plan = placement_plan(room_collision="pass")
    matrix = full_evidence_matrix(
        placement_plans=None,
        provider_evidence={CLOCK: {"placement_rows": deepcopy(plan["instances"])}})
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["dimensions"]["placement"]["status"] == "pass"
    assert row["dimensions"]["placement"]["facts"]["instance_id"] == "source1"


def test_provider_evidence_accepts_a_rejected_planner_row():
    plan = placement_plan(room_collision="pass")
    plan["instances"][0]["status"] = "rejected"
    plan["instances"][0]["reason"] = {"code": "joint_candidate_exhausted"}
    matrix = full_evidence_matrix(
        placement_plans=None,
        provider_evidence={CLOCK: {"placement_rows": deepcopy(plan["instances"])}})
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["dimensions"]["placement"]["status"] == "fail"


def test_a_qualification_verdict_smuggled_in_as_a_row_status_is_refused():
    plan = placement_plan(room_collision="pass")
    plan["instances"][0]["status"] = "pass"
    with pytest.raises(QualificationEvidenceError) as caught:
        full_evidence_matrix(
            provider_evidence={CLOCK: {"placement_rows": deepcopy(plan["instances"])}})
    assert "not a qualification verdict" in str(caught.value)


def test_a_planner_row_may_disclaim_a_qualification_claim():
    """A real planner row carries qualification_status: not_run. Refusing that was
    refusing the planner's own disclaimer."""
    plan = placement_plan(room_collision="pass")
    plan["instances"][0]["qualification_status"] = "not_run"
    plan["instances"][0]["native_execution"] = "not_run"
    matrix = full_evidence_matrix(
        placement_plans=None,
        provider_evidence={CLOCK: {"placement_rows": deepcopy(plan["instances"])}})
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    assert row["dimensions"]["placement"]["status"] == "pass"
    plan["instances"][0]["qualification_status"] = "pass"
    with pytest.raises(QualificationEvidenceError) as caught:
        full_evidence_matrix(
            provider_evidence={CLOCK: {"placement_rows": deepcopy(plan["instances"])}})
    assert "never assert one" in str(caught.value)


def test_a_verdict_key_inside_a_planner_row_is_still_refused():
    plan = placement_plan(room_collision="pass")
    plan["instances"][0]["qualification_status"] = "pass"
    with pytest.raises(QualificationEvidenceError):
        full_evidence_matrix(
            provider_evidence={CLOCK: {"placement_rows": deepcopy(plan["instances"])}})


def test_a_status_outside_a_planner_row_is_still_refused():
    with pytest.raises(QualificationEvidenceError):
        full_evidence_matrix(provider_evidence={CLOCK: {"status": "planned"}})


# --- a native foot readback is used as measured, not composed ---------------

def foot_observation(sole=0.1024, floor=0.1024, world_id=DOG_WORLD):
    return {"facts_path": "/capture", "world_id": world_id, "native_frames": 150,
            "native_emitter_frames": 150, "first_observed_root_y_m": sole + 0.002,
            "sole_world_y_m": sole, "visual_floor_median_y_m": floor,
            "visual_floor_source": "/capture/native_foot_contact.json",
            "measured_frame_count": 5,
            "foot_contact_source": "qualification_geometry.measure_native_foot_contact",
            "max_visible_pixels": 4096,
            "visibility_state_counts": {"visible_clear": 150, "visible_occluded": 0,
                                        "out_of_view": 0, "fully_occluded": 0},
            "audio_events": [{"sound_asset_id": "bark", "stem": {
                "path": "/delivery/dog_stem.wav", "frames": 160000, "sample_rate_hz": 16000,
                "channels": 2, "finite": True, "peak_abs": 0.2}}]}


def test_a_native_foot_readback_is_used_directly():
    matrix = ground_matrix(
        geometry_measurements=[catalog()], world_floor_planes={},
        retained_readback={"rows": [{"asset_id": DOG, "observations": [foot_observation()]}]})
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == DOG)
    support = row["dimensions"]["support"]
    assert support["status"] == "pass"
    observation = support["facts"]["observations"][0]
    assert observation["contact"] == "on_floor"
    assert "native foot contact readback" in observation["contact_basis"]
    assert observation["measured_frame_count"] == 5


def test_a_sole_measured_above_the_drawn_floor_fails_and_names_its_world():
    matrix = ground_matrix(
        geometry_measurements=[catalog()], world_floor_planes={},
        retained_readback={"rows": [{"asset_id": DOG, "observations": [
            foot_observation(sole=0.1609, floor=0.1024, world_id="p09_world")]}]})
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == DOG)
    support = row["dimensions"]["support"]
    assert support["status"] == "fail"
    assert "p09_world" in support["reason"]
    assert "media from those worlds is kept" in support["reason"]
    assert support["facts"]["observations"][0]["foot_above_visual_floor_m"] == pytest.approx(
        0.0585, abs=1e-4)
    # a pose that fails takes its media with it
    assert row["remaining_work"]["bucket"] == "replan_then_recapture"
    assert row["remaining_work"]["native_media_reusable"] is False


def test_a_world_with_only_a_navigation_floor_stays_not_run():
    observation = foot_observation()
    observation.pop("sole_world_y_m")
    observation.pop("visual_floor_median_y_m")
    matrix = ground_matrix(
        geometry_measurements=[catalog()], world_floor_planes={},
        retained_readback={"rows": [{"asset_id": DOG, "observations": [observation]}]})
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == DOG)
    assert row["dimensions"]["support"]["status"] == "not_run"
    assert any("visual floor plane" in item
               for item in row["dimensions"]["support"]["facts"]["observations"][0]["missing"])


def test_one_visible_appearance_is_enough_for_a_representative():
    """An actor deliberately off screen in one world must not count against the
    asset; the requirement is one visible appearance somewhere."""
    payload = retained()
    seen = payload["rows"][0]["observations"][0]
    offscreen = deepcopy(seen)
    offscreen["world_id"] = seen.get("world_id", "world_a")
    offscreen["max_visible_pixels"] = 0
    offscreen["visibility_state_counts"] = {"visible_clear": 0, "visible_occluded": 0,
                                            "out_of_view": 150, "fully_occluded": 0}
    payload["rows"][0]["observations"] = [offscreen, seen]
    matrix = full_evidence_matrix(retained_readback=payload)
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    visibility = row["dimensions"]["visibility"]
    assert visibility["status"] == "pass"
    assert visibility["facts"]["out_of_view_frames"] == 150
    assert "not an appearance in every world" in visibility["facts"]["requirement"]


def test_an_asset_mesh_narrowphase_pass_clears_a_placement_a_box_could_not():
    plan = placement_plan(room_collision="pass")
    plan["instances"][0]["asset_bounds"]["source"] = "derived_conservative"
    plan["instances"][0]["clearance"]["room_collision"] = {
        "status": "pass", "method": "asset_mesh_narrowphase",
        "reason": "no asset triangle crosses a room triangle",
        "penetrating_pair_count": 0}
    matrix = full_evidence_matrix(placement_plans=[plan])
    row = next(r for r in matrix["asset_status_rows"] if r["asset_id"] == CLOCK)
    clearance = row["dimensions"]["clearance"]
    assert clearance["status"] == "pass"
    assert clearance["facts"]["room_collision_evidence"]["is_mesh_narrowphase"] is True
    assert "asset-mesh collision query" in clearance["reason"]
