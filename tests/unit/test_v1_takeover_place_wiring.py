"""Real zero-native regression for static tabletop and wall placements."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from avengine.qa.answerability import MeshHandle, line_of_sight
from avengine.dataset.production_spec import parse_production_config
from avengine.rooms import conditioned_sampler as cs


ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = (
    ROOT
    / "tmp/binding_v1_parallel_20260910/TAKEOVER/T06_PLACE/"
      "attempt_20260910T211358Z_pid781497/support_surface_catalog.json"
)
QUALIFICATION_PATH = ROOT / "examples/dataset/qa_binding_representatives_20260911.json"
BASE_PLAN_PATH = (
    ROOT
    / "tmp/binding_v1_parallel_20260910/TAKEOVER/T03a/"
      "attempt_20260910T185938Z_pid734148/real_registered_retry/"
      "real_registered_plan.json"
)


REGISTRY_PATH = ROOT / "examples/runtime/source_asset_runtime_profiles.json"


def _registry():
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _rigid_instance(entry, registry):
    """The first instance this episode places on a support surface.

    The class comes from the registry record, not from the position in the list
    or from the declared string: since T25 these episodes lead with a speaking
    human anchor, and reading instances[0] handed the sampler a human and an
    empty static request list.
    """
    records = {row["asset_id"]: row for row in registry["assets"]}
    for instance in entry["instances"]:
        record = records.get(instance["asset_id"])
        if record is not None and record["entity_class"] in cs.RIGID:
            return instance
    raise AssertionError(f"{entry['episode_id']} declares no rigid static instance")


def _articulated_instance(entry, registry, *, exclude):
    records = {row["asset_id"]: row for row in registry["assets"]}
    for instance in entry["instances"]:
        if instance["instance_id"] == exclude:
            continue
        record = records.get(instance["asset_id"])
        if record is not None and record["entity_class"] not in cs.RIGID:
            return instance
    raise AssertionError(f"{entry['episode_id']} declares no articulated instance")


def _request(base, qualification, episode_id):
    entry = next(row for row in qualification["episodes"] if row["episode_id"] == episode_id)
    registry = _registry()
    static = _rigid_instance(entry, registry)
    competitor = _articulated_instance(entry, registry, exclude=static["instance_id"])
    selection = entry["sound"]["selection"]
    # Each instance keeps the id the episode gave it: conditioned_sampler looks the
    # placement row up by the actor's entity_instance_id, so renaming the device to
    # source1 would point its row at the wrong actor.
    static_instance_id = static["instance_id"]
    competitor_instance_id = competitor["instance_id"]
    sound_id = selection["preallocated_sound_asset_ids_by_actor"][static_instance_id][0]
    kept = [row for row in entry["instances"]
            if row["instance_id"] in {static_instance_id, competitor_instance_id}]
    formal = next(
        row for row in parse_production_config(qualification).episodes
        if row.request_id == episode_id
    ).to_legacy_request()
    placement = deepcopy(formal["static_source_placement"])
    placement["requests"] = [
        row for row in placement.get("requests", [])
        if row.get("instance_id") == static_instance_id
    ]
    assert placement["requests"], (
        f"{episode_id} has no static placement row for {static_instance_id}")
    request = {
        "episode_id": "t11_test_" + episode_id,
        "room_id": entry["room_id"],
        "room_catalog": str(ROOT / "examples/rooms/packages/catalog.json"),
        "runtime": deepcopy(base["request"]["runtime"]),
        "seed": entry["seed"],
        "sampling_policy": cs.POLICY,
        "frame_count": 150,
        "frame_rate_hz": 15,
        "sample_rate_hz": 16000,
        # positional, so it follows the same order as the instance rows above
        "source_asset_ids": [row["asset_id"] for row in kept],
        "entities": {
            "total_count": 2,
            "silent_count": 1,
            "min_articulated_count": 1,
            # The episode's own instance order is preserved. select_entities names
            # actor slots source1..N by position and also aliases each declared
            # instance_id, so reordering makes one declared id collide with another
            # actor's positional slot and the preallocation lands on the wrong actor.
            "instances": [
                {
                    "instance_id": instance["instance_id"],
                    "asset_id": instance["asset_id"],
                    "source_class": instance["source_class"],
                    "role": "target" if instance["instance_id"] == static_instance_id
                             else "competitor",
                    "speaking": instance["instance_id"] == static_instance_id,
                }
                for instance in kept
            ],
        },
        "profile": {
            "anchor_count": 1,
            "anchor_line_of_sight": "clear",
            "anchor_visibility": "in_fov",
            "competitor_visibility": "in_fov",
            "competitor_motion": "still",
            "speech_motion": "all_still",
            "event_relation": "sequential",
            "reserve_tail_s": 3.0,
            "separation_bin_deg": [15, 90],
            "retry_budget_within_profile": 40,
        },
        "sound_selection": {
            # only the instances this request actually keeps, and only the
            # speaking one: a silent competitor is never preallocated a sound
            "preallocated_sound_asset_ids_by_actor": {
                static_instance_id: [sound_id],
            },
            "sound_class_config": selection["sound_class_config"],
        },
        "camera": {
            "fov_deg": 85.0,
            "height_above_floor_m": 1.55,
            "resolution_hw": [720, 1280],
        },
        "static_source_placement": placement,
    }
    return request


def test_placement_rotation_rejects_reflection_before_quaternion_normalization():
    with pytest.raises(
        cs.CandidateFailure,
        match="static_source_placement_rotation_not_proper",
    ):
        cs._rotation_matrix_to_xyzw(np.diag([1.0, 1.0, -1.0]))


def test_placement_rotation_roundtrip_is_unit_and_reconstructs():
    matrix = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    quaternion = cs._rotation_matrix_to_xyzw(matrix)
    assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    reconstructed = cs._rotation_matrix_from_xyzw(quaternion)
    assert reconstructed == pytest.approx(matrix)


@pytest.fixture(scope="module")
def real_context():
    assert CATALOG_PATH.is_file(), CATALOG_PATH
    assert QUALIFICATION_PATH.is_file(), QUALIFICATION_PATH
    assert BASE_PLAN_PATH.is_file(), BASE_PLAN_PATH
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    qualification = json.loads(QUALIFICATION_PATH.read_text(encoding="utf-8"))
    base = json.loads(BASE_PLAN_PATH.read_text(encoding="utf-8"))
    formal = next(
        row for row in parse_production_config(qualification).episodes
        if row.request_id == "v02_rep_tabletop_pending_01"
    ).to_legacy_request()
    pool_path = Path(formal["sound_pool"])
    if not pool_path.is_absolute():
        pool_path = ROOT / pool_path
    pool_payload = json.loads(pool_path.read_text(encoding="utf-8"))
    sounds = cs.load_conditioned_sound_pool(pool_payload, source_path=pool_path)
    return {
        "catalog": catalog,
        "qualification": qualification,
        "base": base,
        "sounds": sounds,
    }


def test_sampling_candidate_index_rejects_bool_and_negative():
    with pytest.raises(ValueError):
        cs._sampling_candidate_index({"sampling_candidate_index": True})
    with pytest.raises(ValueError):
        cs._sampling_candidate_index({"sampling_candidate_index": -1})


@pytest.mark.parametrize(
    ("episode_id", "surface_kind"),
    [
        ("v02_rep_tabletop_pending_01", "tabletop"),
        ("v02_rep_tabletop_pending_02", "tabletop"),
        ("v02_rep_mount_pending", "wall"),
    ],
)
def test_real_solve_entry_wires_static_support_into_every_frame(
    real_context, episode_id, surface_kind
):
    context = real_context
    request = _request(context["base"], context["qualification"], episode_id)
    registry = json.loads(
        (ROOT / "examples/runtime/source_asset_runtime_profiles.json").read_text(
            encoding="utf-8"
        )
    )
    result = cs.solve_conditioned_episode(
        request=request,
        source_registry=registry,
        sounds=context["sounds"],
        room_id=request["room_id"],
    )
    plan = result["plan"]
    placement_batch = plan["static_source_placements"]
    assert placement_batch["schema"] == "avengine_source_placement_plan_v1"
    assert placement_batch["status"] == "planned"
    placement = next(
        row
        for row in placement_batch["instances"]
        if row["status"] == "planned"
    )
    assert placement["support_identity"]["surface_kind"] == surface_kind
    assert placement["clearance"]["status"] in {"not_run", "partial"}
    room_collision = placement["clearance"].get("room_collision") or {}
    assert room_collision.get("status") == "not_run"

    frames = plan["visual_plan"]["frames"]
    assert len(frames) == 150
    states = [
        next(state for state in frame["actor_states"]
             if state["entity_instance_id"] == placement["instance_id"])
        for frame in frames
    ]
    first = states[0]
    assert all(
        state["root_transform"]["matrix_row_major"]
        == first["root_transform"]["matrix_row_major"]
        for state in states
    )
    assert first["root_transform"]["matrix_row_major"] == placement["root_transform"]["matrix_row_major"]
    assert np.allclose(
        first["root_transform"]["rotation_xyzw"],
        placement["root_transform"]["rotation_xyzw"],
        rtol=0.0,
        atol=1.0e-9,
    )
    assert all(
        state["emitter_transform"] == first["emitter_transform"]
        for state in states
    )
    assert first["emitter_transform"] == placement["emitter_transform"]
    assert all(not state["moving"] for state in states)
    assert all(
        state["support_identity"] == placement["support_identity"]
        for state in states
    )

    ground = next(
        state
        for state in frames[0]["actor_states"]
        if state["entity_instance_id"] != placement["instance_id"]
    )
    floor_y = float(plan["activity_plan"]["selected_floor_height_m"])
    assert abs(float(ground["root_transform"]["translation_m"][1]) - floor_y) <= 0.3
    assert abs(
        float(first["root_transform"]["translation_m"][1])
        - float(ground["root_transform"]["translation_m"][1])
    ) > 0.1
    route_row = next(
        row
        for row in plan["activity_plan"]["actors"]
        if row.get("placement_status") == "planned"
    )
    assert route_row["navigation_authority"] == "support_surface_not_ground"

    camera = plan["visual_plan"]["camera"]
    assert camera["motion"] == "static"
    geometry = (plan["resources"]["room_package"] or {})["static_geometry"]
    mesh = MeshHandle.from_paths(geometry["vertices"], geometry["triangles"])
    emitter = first["emitter_transform"]["position_m"]
    camera_position = camera["position_m"]
    assert line_of_sight(mesh, camera_position, emitter) == "clear"
    delta = np.asarray(emitter, dtype=float) - np.asarray(camera_position, dtype=float)
    forward = np.asarray(camera["basis"]["forward"], dtype=float)
    right = np.asarray(camera["basis"]["right"], dtype=float)
    up = np.asarray(camera["basis"]["up"], dtype=float)
    depth = float(np.dot(forward, delta))
    side = float(np.dot(right, delta))
    vertical = float(np.dot(up, delta))
    tangent = math.tan(math.radians(camera["horizontal_fov_deg"]) / 2.0)
    aspect = camera["resolution_hw"][1] / camera["resolution_hw"][0]
    assert depth > 0.1
    assert abs(side) < depth * tangent
    assert abs(vertical) < depth * tangent / aspect
    assert plan["evidence_status"] == {
        "native_visual": "not_run",
        "native_audio": "not_run",
        "qa_validity": "not_run",
    }
